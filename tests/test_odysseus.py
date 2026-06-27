from __future__ import annotations

import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

from planning.trajectory_rollout import CandidateEvaluation
from runtime.core.cognition import CognitiveObservation, OccupancyGridSpec
from runtime.core.contracts import PlannerCommand, RuntimeStamp
from runtime.core.navigation_intelligence import Pose2DState
from runtime.core.odysseus import (
    ODYSSEUS_CAUSE_LABELS,
    ODYSSEUS_FEATURE_SIZE,
    OdysseusAttributionConfig,
    OdysseusCausalAttributor,
    OdysseusOutcome,
    OdysseusNavigator,
    OdysseusShadowRunner,
    OdysseusWorldMemory,
    RolloutCandidateRecord,
    build_causal_feature_vector,
    load_causal_sample,
    load_odysseus_checkpoint,
    make_causal_sample,
    odysseus_attribution_loss,
    save_causal_sample,
    save_odysseus_checkpoint,
    stack_causal_samples,
)
from runtime.core.odysseus.attribution import torch


REPO_ROOT = Path(__file__).resolve().parents[1]


def _observation() -> CognitiveObservation:
    return CognitiveObservation(
        stamp=RuntimeStamp(10.0, "map", "test"),
        pose=Pose2DState(10.0, 1.0, 0.5, 0.1, "test"),
        goal_world_xy=np.asarray([5.0, 1.0], dtype=np.float32),
        local_occupancy=np.zeros((32, 32), dtype=np.float32),
        global_occupancy=np.zeros((40, 40), dtype=np.float32),
        local_grid_spec=OccupancyGridSpec(0.1, 0.0, -1.6, "base_link"),
        global_grid_spec=OccupancyGridSpec(0.2, -2.0, -2.0, "map"),
        route_world_xy=np.asarray([[1.0, 0.5], [2.0, 0.7], [5.0, 1.0]], dtype=np.float32),
        previous_command=PlannerCommand(RuntimeStamp(9.9, "base_link", "test"), 0.2, 0.1, "prev", "test"),
        localization_uncertainty=0.15,
        sensor_ages_sec={"depth": 0.05, "pose": 0.04},
    )


def _candidates() -> tuple[RolloutCandidateRecord, ...]:
    return (
        RolloutCandidateRecord.from_evaluation(
            CandidateEvaluation(0.24, 0.0, True, -0.8, "ok", 1.4),
            candidate_id="straight",
            selected=True,
        ),
        RolloutCandidateRecord.from_evaluation(
            CandidateEvaluation(0.16, 0.45, True, -0.3, "ok", 2.2),
            candidate_id="left",
        ),
        RolloutCandidateRecord.from_evaluation(
            CandidateEvaluation(0.24, -0.45, False, float("inf"), "collision", 0.25),
            candidate_id="right",
        ),
    )


class OdysseusTests(unittest.TestCase):
    def test_feature_vector_and_causal_sample_round_trip(self):
        candidates = _candidates()
        features = build_causal_feature_vector(
            _observation(),
            candidates[0],
            candidates,
            semantic_forward_m=1.7,
            semantic_lateral_m=-0.2,
            progress_distance_m=3.5,
        )
        self.assertEqual(features.shape, (ODYSSEUS_FEATURE_SIZE,))
        self.assertTrue(np.isfinite(features).all())

        runner = OdysseusShadowRunner()
        trace = runner.observe(_observation(), selected_candidate=candidates[0], candidate_records=candidates).trace
        outcome = OdysseusOutcome(
            progress_delta_m=-0.2,
            collision=False,
            stuck=True,
            safety_override=True,
            localization_diverged=False,
            stale_sensor=False,
            final_goal_distance_m=3.8,
            failure_cause="local_minimum",
            severity=0.7,
        )
        sample = make_causal_sample(trace, outcome)
        with tempfile.TemporaryDirectory() as directory:
            path = save_causal_sample(sample, Path(directory) / "odysseus_sample.npz")
            loaded = load_causal_sample(path)
        self.assertEqual(ODYSSEUS_CAUSE_LABELS[loaded.cause_index], "local_minimum")
        self.assertEqual(loaded.feature_vector.shape, (ODYSSEUS_FEATURE_SIZE,))
        self.assertAlmostEqual(float(loaded.outcome_targets[7]), 0.7, places=5)

    def test_shadow_episode_closure_records_owned_sample(self):
        candidates = _candidates()
        with tempfile.TemporaryDirectory() as directory:
            runner = OdysseusShadowRunner(dataset_directory=directory)
            result = runner.observe(
                _observation(),
                selected_candidate=candidates[0],
                candidate_records=candidates,
                trace_id="episode-a",
            )
            closed = runner.close_episode(
                result.trace.trace_id,
                OdysseusOutcome(0.4, False, False, False, False, False, 2.1, "success", 0.0),
            )
            self.assertIsNotNone(closed.sample_path)
            self.assertTrue(closed.sample_path.exists())
            self.assertEqual(load_causal_sample(closed.sample_path).metadata["trace_id"], "episode-a")


    def test_persistent_navigator_closes_outcomes_and_adapts(self):
        first = _observation()
        second = CognitiveObservation(
            stamp=RuntimeStamp(10.4, "map", "test"),
            pose=Pose2DState(10.4, 1.01, 0.5, 0.1, "test"),
            goal_world_xy=np.asarray([5.0, 1.0], dtype=np.float32),
            local_occupancy=np.zeros((32, 32), dtype=np.float32),
            global_occupancy=np.zeros((40, 40), dtype=np.float32),
            local_grid_spec=OccupancyGridSpec(0.1, 0.0, -1.6, "base_link"),
            global_grid_spec=OccupancyGridSpec(0.2, -2.0, -2.0, "map"),
            route_world_xy=np.asarray([[1.01, 0.5], [2.0, 0.7], [5.0, 1.0]], dtype=np.float32),
            previous_command=PlannerCommand(RuntimeStamp(10.3, "base_link", "test"), 0.24, 0.0, "prev", "test"),
            localization_uncertainty=0.15,
            sensor_ages_sec={"depth": 0.05, "pose": 0.04},
        )
        evaluations = [
            CandidateEvaluation(0.24, 0.0, True, -0.8, "ok", 1.4),
            CandidateEvaluation(0.12, 0.55, True, -0.2, "ok", 2.5),
            CandidateEvaluation(0.24, -0.55, False, float("inf"), "collision", 0.2),
        ]
        deterministic = PlannerCommand(RuntimeStamp(10.0, "base_link", "rollout"), 0.24, 0.0, "rollout_drive", "baseline")
        with tempfile.TemporaryDirectory() as directory:
            navigator = OdysseusNavigator(shadow_runner=OdysseusShadowRunner(dataset_directory=directory), blocked_window=2)
            first_decision = navigator.decide(
                first,
                evaluations,
                deterministic_command=deterministic,
                goal_distance_m=4.0,
                semantic_forward_m=2.5,
                progress_distance_m=4.0,
            )
            self.assertTrue(first_decision.command.mode.startswith("odysseus_"))
            second_decision = navigator.decide(
                second,
                evaluations,
                deterministic_command=deterministic,
                goal_distance_m=4.05,
                semantic_forward_m=0.7,
                progress_distance_m=4.05,
                nav_safety_action="STOP",
                nav_motion_state="OK",
            )
            self.assertEqual(second_decision.mode, "recover")
            self.assertTrue(second_decision.closed_sample_path)
            loaded = load_causal_sample(second_decision.closed_sample_path)
            self.assertIn(ODYSSEUS_CAUSE_LABELS[loaded.cause_index], {"thin_obstacle_missed", "overconfident_clearance"})
            self.assertGreaterEqual(navigator.closed_samples, 1)



    def test_missing_trace_does_not_disable_next_odysseus_decision(self):
        candidates = [
            CandidateEvaluation(0.24, 0.0, True, -0.8, "ok", 1.4),
            CandidateEvaluation(0.12, 0.55, True, -0.2, "ok", 2.5),
        ]
        navigator = OdysseusNavigator(shadow_runner=OdysseusShadowRunner())
        first = navigator.decide(
            _observation(),
            [CandidateEvaluation(0.24, 0.0, True, -0.8, "ok", 1.4)],
            deterministic_command=PlannerCommand(RuntimeStamp(10.0, "base_link", "rollout"), 0.24, 0.0, "rollout_drive", "baseline"),
            goal_distance_m=4.0,
            semantic_forward_m=2.5,
            progress_distance_m=4.0,
        )
        navigator.shadow_runner._open_traces.pop(first.trace_id)
        second = CognitiveObservation(
            stamp=RuntimeStamp(10.2, "map", "test"),
            pose=Pose2DState(10.2, 1.05, 0.5, 0.1, "test"),
            goal_world_xy=np.asarray([5.0, 1.0], dtype=np.float32),
            local_occupancy=np.zeros((32, 32), dtype=np.float32),
            global_occupancy=np.zeros((40, 40), dtype=np.float32),
            local_grid_spec=OccupancyGridSpec(0.1, 0.0, -1.6, "base_link"),
            global_grid_spec=OccupancyGridSpec(0.2, -2.0, -2.0, "map"),
            route_world_xy=np.asarray([[1.05, 0.5], [2.0, 0.7], [5.0, 1.0]], dtype=np.float32),
            previous_command=PlannerCommand(RuntimeStamp(10.1, "base_link", "test"), 0.24, 0.0, "prev", "test"),
            localization_uncertainty=0.15,
            sensor_ages_sec={"depth": 0.05, "pose": 0.04},
        )
        decision = navigator.decide(
            second,
            candidates,
            deterministic_command=PlannerCommand(RuntimeStamp(10.2, "base_link", "rollout"), 0.24, 0.0, "rollout_drive", "baseline"),
            goal_distance_m=3.9,
            semantic_forward_m=2.5,
            progress_distance_m=3.9,
        )
        self.assertTrue(decision.command.mode.startswith("odysseus_"))
        self.assertNotEqual(decision.reason, "")

    def test_world_memory_persists_no_go_and_changes_second_run_choice(self):
        evaluations = [
            CandidateEvaluation(0.24, 0.0, True, -0.8, "ok", 1.4),
            CandidateEvaluation(0.12, 0.55, True, -0.2, "ok", 2.5),
        ]
        deterministic = PlannerCommand(RuntimeStamp(10.0, "base_link", "rollout"), 0.24, 0.0, "rollout_drive", "baseline")
        with tempfile.TemporaryDirectory() as directory:
            memory_path = Path(directory) / "memory.json"
            memory = OdysseusWorldMemory()
            memory.record_outcome(
                pose_xy=(1.25, 0.55),
                command_mode="odysseus_advance",
                action_bucket="drive_straight",
                outcome=OdysseusOutcome(-0.2, False, True, True, False, False, 4.0, "local_minimum", 0.9),
                time_sec=10.5,
            )
            memory.save(memory_path)
            loaded = OdysseusWorldMemory.load(memory_path)
            self.assertGreaterEqual(len(loaded.remembered_no_go_points()), 1)

            navigator = OdysseusNavigator(
                shadow_runner=OdysseusShadowRunner(),
                memory_path=str(memory_path),
            )
            decision = navigator.decide(
                _observation(),
                evaluations,
                deterministic_command=deterministic,
                goal_distance_m=4.0,
                semantic_forward_m=2.5,
                progress_distance_m=4.0,
            )
            self.assertNotEqual(decision.selected_candidate_id, "rollout_000")

    @unittest.skipIf(torch is None, "PyTorch is not installed")
    def test_mlp_loss_and_checkpoint_round_trip(self):
        candidates = _candidates()
        runner = OdysseusShadowRunner()
        trace = runner.observe(_observation(), selected_candidate=candidates[0], candidate_records=candidates).trace
        samples = [
            make_causal_sample(trace, OdysseusOutcome(0.5, False, False, False, False, False, 2.0, "success", 0.0)),
            make_causal_sample(trace, OdysseusOutcome(-0.1, False, True, True, False, False, 2.8, "local_minimum", 0.8)),
        ]
        batch = stack_causal_samples(samples)
        model = OdysseusCausalAttributor(OdysseusAttributionConfig(hidden_size=32, dropout=0.0))
        outputs = model(batch["features"])
        self.assertEqual(outputs["cause_logits"].shape, (2, len(ODYSSEUS_CAUSE_LABELS)))
        loss, components = odysseus_attribution_loss(outputs, batch["cause"], batch["outcomes"])
        loss.backward()
        self.assertIn("cause", components)
        with tempfile.TemporaryDirectory() as directory:
            path = save_odysseus_checkpoint(model, Path(directory) / "odysseus.pt", epoch=1)
            loaded, checkpoint = load_odysseus_checkpoint(path)
        loaded_outputs = loaded(batch["features"])
        self.assertEqual(checkpoint["epoch"], 1)
        self.assertEqual(loaded_outputs["cause_logits"].shape, outputs["cause_logits"].shape)

    @unittest.skipIf(torch is None, "PyTorch is not installed")
    def test_training_cli_one_epoch_smoke(self):
        candidates = _candidates()
        runner = OdysseusShadowRunner()
        trace = runner.observe(_observation(), selected_candidate=candidates[0], candidate_records=candidates).trace
        with tempfile.TemporaryDirectory() as directory:
            dataset = Path(directory) / "dataset"
            dataset.mkdir()
            for index, cause in enumerate(("success", "local_minimum")):
                outcome = OdysseusOutcome(
                    0.4 if cause == "success" else -0.2,
                    False,
                    cause != "success",
                    cause != "success",
                    False,
                    False,
                    2.0 + index,
                    cause,
                    0.0 if cause == "success" else 0.75,
                )
                save_causal_sample(make_causal_sample(trace, outcome), dataset / f"odysseus_{index}.npz")
            output = Path(directory) / "model.pt"
            completed = subprocess.run(
                [
                    sys.executable,
                    str(REPO_ROOT / "tools" / "train_odysseus_attributor.py"),
                    str(dataset),
                    "--epochs",
                    "1",
                    "--batch-size",
                    "2",
                    "--output",
                    str(output),
                ],
                cwd=REPO_ROOT,
                check=True,
                text=True,
                capture_output=True,
            )
            self.assertIn("saved Odysseus checkpoint", completed.stdout)
            self.assertTrue(output.exists())


if __name__ == "__main__":
    unittest.main()
