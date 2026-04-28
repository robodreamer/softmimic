# test.py

"""
Script to test environments, log data, and optionally play a checkpoint
of an RL agent from RSL-RL.

Launch Isaac Sim Simulator first.
"""

import argparse
import os
import torch
from datetime import datetime
import inspect
import re 
import numpy as np
from typing import Dict, List, Tuple, Optional

from isaaclab.app import AppLauncher
import gymnasium as gym

# local imports
import cli_args  # isort: skip


# add argparse arguments
parser = argparse.ArgumentParser(description="Test an environment or play a trained RL agent with RSL-RL.")
parser.add_argument("--video", action="store_true", default=False, help="Record videos during testing/playback.")
parser.add_argument("--video_length", type=int, default=200, help="Length of the recorded video (in steps).")
parser.add_argument(
    "--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O operations."
)
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument("--use_zero_action", action="store_true", default=False, help="Force using zero actions even if a checkpoint is specified.")
parser.add_argument("--load_pulse_run", type=str, default=None, help="Run name to load PULSE model from.")
parser.add_argument("--csv_file", type=str, default=None, help="Path to CSV file for motion reference.")
parser.add_argument("--csv_start", type=float, default=0.0, help="Start time for the CSV file.")
parser.add_argument("--csv_end", type=float, default=0.0, help="End time for the CSV file (0 means no limit).")

# Add WandB arguments for loading checkpoints
parser.add_argument("--load_run_wandb_id", type=str, default=None, help="WandB run ID to load model from.")
parser.add_argument("--project_entity", type=str, default=None, help="WandB project entity (e.g., 'your_username').")
parser.add_argument("--disable_wandb_logging", action="store_true", default=False, help="Disable logging evaluation results to WandB.")

# Add evaluation mode arguments
parser.add_argument("--eval_mode", action="store_true", default=False, help="Enable evaluation mode to test multiple checkpoints.")
parser.add_argument("--eval_checkpoint_num", type=int, default=None, help="Evaluate a single specific checkpoint number (e.g., 8000).")
parser.add_argument("--eval_checkpoint_interval", type=int, default=2000, help="Interval between checkpoints to evaluate (e.g., 2000 for model_0.pt, model_2000.pt, etc.).")
parser.add_argument("--eval_max_checkpoint", type=int, default=None, help="Maximum checkpoint number to evaluate (if None, will find the highest available).")
parser.add_argument("--eval_duration", type=float, default=30.0, help="Duration in seconds to evaluate each checkpoint.")
parser.add_argument("--eval_num_episodes", type=int, default=None, help="Number of episodes to evaluate per checkpoint (overrides eval_duration if specified).")
parser.add_argument("--eval_record_video", action="store_true", default=False, help="Record videos during evaluation and log to WandB.")
parser.add_argument("--eval_video_length", type=int, default=1500, help="Length of evaluation videos in steps.")
parser.add_argument("--eval_video_fps", type=int, default=50, help="FPS for evaluation videos.")

# append RSL-RL cli arguments (for policy loading)
cli_args.add_rsl_rl_args(parser)
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

# always enable cameras if recording video
if args_cli.video or args_cli.eval_record_video:
    args_cli.enable_cameras = True

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

# Isaac Lab imports
from isaaclab.envs import DirectMARLEnv, multi_agent_to_single_agent
from isaaclab.utils.dict import print_dict
from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlVecEnvWrapper, export_policy_as_jit, export_policy_as_onnx
from isaaclab_tasks.utils import get_checkpoint_path, parse_env_cfg
import isaaclab.terrains as terrain_gen
from isaaclab.managers import DatasetExportMode, RecorderTerm, RecorderTermCfg

# RSL-RL imports (needed for policy loading)
from rsl_rl.runners import OnPolicyRunner

# Import extensions to set up environment tasks
import softmimic_gym.tasks

class EvaluationStats:
    """Class to track evaluation statistics for a single checkpoint."""
    
    def __init__(self):
        self.episode_returns = []
        self.episode_lengths = []
        self.episode_count = 0
        self.total_steps = 0
        self.mean_rewards = {}  # Store mean reward components
        
    def add_episode(self, episode_return: float, episode_length: int):
        """Add completed episode statistics."""
        self.episode_returns.append(episode_return)
        self.episode_lengths.append(episode_length)
        self.episode_count += 1
        self.total_steps += episode_length
        
    def get_summary(self) -> Dict:
        """Get summary statistics."""
        if not self.episode_returns:
            return {
                "mean_return": 0.0,
                "std_return": 0.0,
                "mean_length": 0.0,
                "std_length": 0.0,
                "episode_count": 0,
                "total_steps": 0,
            }
        
        return {
            "mean_return": np.mean(self.episode_returns),
            "std_return": np.std(self.episode_returns),
            "min_return": np.min(self.episode_returns),
            "max_return": np.max(self.episode_returns),
            "mean_length": np.mean(self.episode_lengths),
            "std_length": np.std(self.episode_lengths),
            "min_length": np.min(self.episode_lengths),
            "max_length": np.max(self.episode_lengths),
            "episode_count": self.episode_count,
            "total_steps": self.total_steps,
        }


def find_available_checkpoints(run, checkpoint_interval: int, max_checkpoint: Optional[int] = None) -> List[int]:
    """Find all available checkpoints at specified intervals."""
    available_checkpoints = []
    
    # Get all .pt files from the run
    pt_files = [f.name for f in run.files() if f.name.endswith('.pt')]
    
    # Extract checkpoint numbers as integers
    checkpoint_numbers = []
    for filename in pt_files:
        # Look for pattern like "model_XXXX.pt"
        match = re.search(r'model_(\d+)\.pt', filename)
        if match:
            checkpoint_numbers.append(int(match.group(1)))  # Convert to int for proper numeric sorting
    
    # Sort numerically and filter by interval and max_checkpoint
    checkpoint_numbers.sort()  # Sort integers in ascending order
    for num in checkpoint_numbers:
        if num % checkpoint_interval == 0:
            if max_checkpoint is None or num <= max_checkpoint:
                available_checkpoints.append(num)
    
    print(f"[INFO] Found checkpoint numbers: {sorted(checkpoint_numbers)}")
    print(f"[INFO] Found {len(available_checkpoints)} checkpoints at interval {checkpoint_interval}: {available_checkpoints}")
    return available_checkpoints


def load_checkpoint_from_wandb(run, checkpoint_num: int, download_dir: str) -> Tuple[Optional[str], Optional[str]]:
    """Download and return paths to checkpoint and JIT files."""
    checkpoint_filename = f"model_{checkpoint_num}.pt"
    jit_filename = f"model_{checkpoint_num}.jit"
    
    checkpoint_path = None
    jit_path = None
    
    # Download checkpoint file
    try:
        run.file(checkpoint_filename).download(root=download_dir, replace=True)
        checkpoint_path = os.path.join(download_dir, checkpoint_filename)
        if os.path.exists(checkpoint_path):
            print(f"[INFO] Downloaded checkpoint: {checkpoint_path}")
        else:
            print(f"[WARNING] Failed to download checkpoint: {checkpoint_filename}")
            return None, None
    except Exception as e:
        print(f"[ERROR] Error downloading checkpoint {checkpoint_filename}: {e}")
        return None, None
    
    # Try to download JIT file (optional)
    try:
        run.file(jit_filename).download(root=download_dir, replace=True)
        jit_path = os.path.join(download_dir, jit_filename)
        if os.path.exists(jit_path):
            print(f"[INFO] Downloaded JIT file: {jit_path}")
        else:
            jit_path = None
    except Exception as e:
        print(f"[INFO] JIT file not available or failed to download: {e}")
        jit_path = None
    
    return checkpoint_path, jit_path


def create_policy_from_checkpoint(checkpoint_path: str, agent_cfg, env, args_cli):
    """Create and load policy from checkpoint."""
    print(f"[INFO] Loading policy from: {checkpoint_path}")
    
    # Import necessary classes
    import rsl_rl.runners.on_policy_runner
    from softmimic_gym.rsl_rl_extensions.runners import JitSavingOnPolicyRunner as OnPolicyRunner

    # Create runner based on policy type
    runner_log_dir = None
    
    runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=runner_log_dir, device=agent_cfg.device)
    runner.load(checkpoint_path)

    # Get inference policy
    policy = runner.get_inference_policy(device=env.unwrapped.device)

    return policy, runner


def run_evaluation(
    env,
    policy,
    eval_duration: float,
    eval_num_episodes: Optional[int],
    env_cfg,
    record_video: bool = False,
    video_length: int = 500,
    video_fps: int = 50,
    video_save_path: Optional[str] = None,
) -> Tuple[EvaluationStats, Optional[str]]:
    """Run evaluation for specified duration or number of episodes."""
    stats = EvaluationStats()
    
    # Calculate time limits
    dt = env_cfg.decimation * env_cfg.sim.dt
    if eval_num_episodes is not None:
        max_steps = float('inf')  # No step limit, just episode count
        max_episodes = eval_num_episodes
        print(f"[INFO] Running evaluation for {eval_num_episodes} episodes")
    else:
        max_steps = int(eval_duration / dt)
        max_episodes = float('inf')  # No episode limit, just time
        print(f"[INFO] Running evaluation for {eval_duration}s ({max_steps} steps)")
    
    # Setup video recording if requested
    video_writer = None
    video_frames = []
    video_file_path = None
    
    if record_video and video_save_path:
        import imageio
        video_file_path = video_save_path
        print(f"[INFO] Recording video to: {video_file_path}")
    
    # Reset environment and tracking variables
    with torch.inference_mode():
        env.unwrapped.command_manager.get_term("upper_body_joints").motion_count[:] = 250 # 5 seconds into the clip
        env.reset()
        # env.unwrapped._reset_idx(torch.arange(env.num_envs, device=env.unwrapped.device))
        # reset the motion lib times
    obs, info = env.get_observations()
    # obs, info = env.reset()
    timestep = 0
    current_episode_returns = torch.zeros(env.num_envs, device=env.unwrapped.device)
    current_episode_lengths = torch.zeros(env.num_envs, device=env.unwrapped.device, dtype=torch.int)
    
    print(f"[INFO] Starting evaluation with {env.num_envs} environments...")
    
    while timestep < max_steps and stats.episode_count < max_episodes:
        with torch.inference_mode():
            # Get actions from policy
            if policy is not None:
                actions = policy(obs)
            else:
                actions = torch.zeros(env.num_envs, env.num_actions, device=env.unwrapped.device, dtype=torch.float32)
            
            # Step environment
            next_obs_tuple = env.step(actions)
            obs = next_obs_tuple[0]
            rewards = next_obs_tuple[1]
            dones = next_obs_tuple[2]
            info = next_obs_tuple[3]
            
            # Record video frame if requested and within video length
            if record_video and len(video_frames) < video_length:
                try:
                    frame = env.unwrapped.render()
                    if frame is not None:
                        video_frames.append(frame)
                except Exception as e:
                    print(f"[WARNING] Failed to capture video frame: {e}")
            
            # Update episode tracking
            current_episode_returns += rewards
            current_episode_lengths += 1
            
            # Handle episode completions
            if torch.any(dones):
                done_indices = torch.where(dones)[0]
                for idx in done_indices:
                    episode_return = current_episode_returns[idx].item()
                    episode_length = current_episode_lengths[idx].item()
                    stats.add_episode(episode_return, episode_length)
                    
                    # Reset tracking for completed episodes
                    current_episode_returns[idx] = 0.0
                    current_episode_lengths[idx] = 0
            
            timestep += 1
            
            # Progress reporting
            if timestep % 500 == 0:
                print(f"[INFO] Evaluation step {timestep}/{max_steps}, Episodes completed: {stats.episode_count}")
    
    # Save video if frames were recorded
    if record_video and video_frames and video_file_path:
        try:
            import imageio
            print(f"[INFO] Saving video with {len(video_frames)} frames to {video_file_path}")
            with imageio.get_writer(video_file_path, fps=video_fps) as writer:
                for frame in video_frames:
                    writer.append_data(frame)
            print(f"[INFO] Video saved successfully")
        except Exception as e:
            print(f"[ERROR] Failed to save video: {e}")
            video_file_path = None
    
    print(f"[INFO] Evaluation completed. Total episodes: {stats.episode_count}, Total steps: {timestep}")
    return stats, video_file_path


def run_evaluation_mode(args_cli):
    """Main evaluation mode function."""
    if args_cli.project_entity is None or args_cli.load_run_wandb_id is None:
        raise ValueError("Evaluation mode requires --project_entity and --load_run_wandb_id")
    
    print("[INFO] Starting evaluation mode...")
    
    # Get source run info first
    import wandb
    api = wandb.Api()
    wandb_project = "isaaclab"
    source_run = api.run(f"{args_cli.project_entity}/{wandb_project}/{args_cli.load_run_wandb_id}")
    
    if not args_cli.disable_wandb_logging:
        # Create a SEPARATE evaluation run linked to the original
        eval_run_name = f"EVAL_{source_run.name}_{datetime.now().strftime('%H%M%S')}"
        print(f"[INFO] Creating separate evaluation run: {eval_run_name}")
        
        wandb.init(
            project=wandb_project,
            entity=args_cli.project_entity,
            name=eval_run_name,
            tags=["evaluation", f"source_run_{args_cli.load_run_wandb_id}"],
            config={
                "eval_source_run_id": args_cli.load_run_wandb_id,
                "eval_source_run_name": source_run.name,
                "eval_checkpoint_interval": args_cli.eval_checkpoint_interval,
                "eval_duration": args_cli.eval_duration,
                "eval_num_episodes": args_cli.eval_num_episodes,
                "eval_record_video": args_cli.eval_record_video,
                "eval_video_length": args_cli.eval_video_length,
                "eval_video_fps": args_cli.eval_video_fps,
                "eval_task": args_cli.task,
                "eval_num_envs": args_cli.num_envs,
                "eval_timestamp": datetime.now().isoformat(),
            },
            notes=f"Evaluation of checkpoints from training run {args_cli.load_run_wandb_id}"
        )
        print(f"[INFO] Evaluation WandB run URL: {wandb.run.url}")
    else:
        print("[INFO] WandB logging is disabled by user.")
    
    
    print(f"[INFO] Original training run URL: {source_run.url}")
    
    # --- MODIFIED: Find available checkpoints ---
    if args_cli.eval_checkpoint_num is not None:
        # If a single checkpoint is specified, just use that one
        available_checkpoints = [args_cli.eval_checkpoint_num]
        print(f"[INFO] Evaluating a single specified checkpoint: {available_checkpoints[0]}")
    else:
        # Otherwise, find all checkpoints based on interval
        available_checkpoints = find_available_checkpoints(
            source_run,
            args_cli.eval_checkpoint_interval,
            args_cli.eval_max_checkpoint
        )

    if not available_checkpoints:
        print("[ERROR] No checkpoints found matching criteria")
        wandb.finish()
        return
    
    # Setup environment (similar to main function)
    env_cfg = parse_env_cfg(
        args_cli.task, device=args_cli.device, num_envs=args_cli.num_envs, use_fabric=not args_cli.disable_fabric
    )
    agent_cfg: RslRlOnPolicyRunnerCfg = cli_args.parse_rsl_rl_cfg(args_cli.task, args_cli)
    
    # Enable cameras if recording video
    if args_cli.eval_record_video:
        # We need to enable cameras for video recording
        print("[INFO] Enabling cameras for video recording")
    
    # Disable recorders
    env_cfg.recorders = None

    # Apply CSV configurations if needed
    if args_cli.csv_file is not None:
        if args_cli.csv_file == "LAFAN1_ALL":
            env_cfg.commands.upper_body_joints.demo_start_range = [[args_cli.csv_start, args_cli.csv_start] for _ in env_cfg.commands.upper_body_joints.demo_recording_path]
            env_cfg.commands.upper_body_joints.demo_random_init = False
        else:
            env_cfg.commands.upper_body_joints.demo_recording_path = [args_cli.csv_file]
            env_cfg.commands.upper_body_joints.demo_start_range = [[args_cli.csv_start, args_cli.csv_start]]
            env_cfg.commands.upper_body_joints.demo_random_init = False
        if args_cli.csv_end > 0.0:
            env_cfg.commands.upper_body_joints.resampling_time_range = [args_cli.csv_end-args_cli.csv_start, args_cli.csv_end-args_cli.csv_start]
    
    env_cfg.commands.upper_body_joints.demo_random_init = False
    # env_cfg.commands.upper_body_joints.demo_zero_reset = True
    env_cfg.events.reset_robot.params["jpos_range"] = (0.0, 0.0)
    
    # Set episode length for evaluation
    eval_duration = args_cli.eval_duration if args_cli.eval_num_episodes is None else 300.0  # Default long episode for episode-based eval
    env_cfg.episode_length_s = eval_duration
    
    # Setup terrain for consistent evaluation
    if hasattr(env_cfg, "scene") and hasattr(env_cfg.scene, "terrain") and env_cfg.scene.terrain is not None:
        if hasattr(env_cfg.scene.terrain, "terrain_generator"):
            terrain_gen_cfg = env_cfg.scene.terrain.terrain_generator
            if hasattr(terrain_gen_cfg, 'num_rows'):
                terrain_gen_cfg.num_rows = 1 #min(8, args_cli.num_envs // 8)  # Limit to max 8 rows for evaluation
            if hasattr(terrain_gen_cfg, 'num_cols'):
                terrain_gen_cfg.num_cols = 1
            if hasattr(terrain_gen_cfg, 'size'):
                terrain_gen_cfg.size = [2.0, 2.0]
            if hasattr(terrain_gen_cfg, 'curriculum'):
                terrain_gen_cfg.curriculum = False
            terrain_gen_cfg.sub_terrains = {
                "flat": terrain_gen.MeshPlaneTerrainCfg(proportion=1.0),
            }
    
    # Create environment with appropriate render mode
    render_mode = "rgb_array" if args_cli.eval_record_video else None
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode=render_mode)
    
    # Convert to single-agent if needed
    if isinstance(env.unwrapped, DirectMARLEnv):
        print("[INFO] Converting Multi-Agent environment to Single-Agent.")
        env = multi_agent_to_single_agent(env)
    
    # Wrap for RSL-RL compatibility
    env = RslRlVecEnvWrapper(env)
    
    # Create download directory
    download_dir = os.path.join("logs", "eval_checkpoints", args_cli.load_run_wandb_id)
    os.makedirs(download_dir, exist_ok=True)
    
    # Create video directory if recording videos
    video_dir = None
    if args_cli.eval_record_video:
        video_dir = os.path.join(download_dir, "videos")
        os.makedirs(video_dir, exist_ok=True)
    
    # Evaluate each checkpoint
    all_results = {}
    
    for i, checkpoint_num in enumerate(available_checkpoints):
        print(f"\n[INFO] Evaluating checkpoint {i+1}/{len(available_checkpoints)}: model_{checkpoint_num}.pt")
        
        # Download checkpoint
        checkpoint_path, jit_path = load_checkpoint_from_wandb(source_run, checkpoint_num, download_dir)
        
        if checkpoint_path is None:
            print(f"[WARNING] Skipping checkpoint {checkpoint_num} due to download failure")
            continue
        
        # Load policy
        policy, runner = create_policy_from_checkpoint(checkpoint_path, agent_cfg, env, args_cli)
        
        # Setup video recording for this checkpoint
        video_file_path = None
        if args_cli.eval_record_video and video_dir:
            video_file_path = os.path.join(video_dir, f"eval_checkpoint_{checkpoint_num}.mp4")
        
        # Run evaluation
        stats, saved_video_path = run_evaluation(
            env,
            policy,
            args_cli.eval_duration,
            args_cli.eval_num_episodes,
            env_cfg,
            record_video=args_cli.eval_record_video,
            video_length=args_cli.eval_video_length,
            video_fps=args_cli.eval_video_fps,
            video_save_path=video_file_path,
        )
        
        # Get summary statistics
        summary = stats.get_summary()
        all_results[checkpoint_num] = summary
        
        # Use checkpoint number as step - clean and simple since this is a separate run
        eval_step = checkpoint_num
        
        if not args_cli.disable_wandb_logging:
            # Log evaluation metrics with clear, simple names
            eval_log_data = {
                # Main metrics for easy plotting
                "mean_return": summary["mean_return"],
                "std_return": summary["std_return"], 
                "mean_episode_length": summary["mean_length"],
                "episode_count": summary["episode_count"],
                
                # Additional detailed metrics
                "min_return": summary["min_return"],
                "max_return": summary["max_return"],
                "min_episode_length": summary["min_length"], 
                "max_episode_length": summary["max_length"],
                "total_steps": summary["total_steps"],
                
                # Progress tracking
                "checkpoints_evaluated": i + 1,
                "checkpoint_progress": (i + 1) / len(available_checkpoints),
            }
            
            # Log video to WandB if available
            if saved_video_path and os.path.exists(saved_video_path):
                try:
                    print(f"[INFO] Uploading video to WandB: {saved_video_path}")
                    video_wandb = wandb.Video(saved_video_path, fps=args_cli.eval_video_fps, format="mp4")
                    eval_log_data[f"videos/checkpoint_{checkpoint_num}"] = video_wandb
                    eval_log_data["video_latest"] = video_wandb  # Always show latest video
                except Exception as e:
                    print(f"[WARNING] Failed to upload video to WandB: {e}")
            
            # Log the data - simple and clean
            try:
                print(f"[INFO] Logging evaluation data with step {eval_step}")
                wandb.log(eval_log_data, step=eval_step) 
                print(f"[INFO] Successfully logged checkpoint {checkpoint_num} evaluation data")
                
            except Exception as e:
                print(f"[ERROR] Failed to log evaluation data: {e}")
        
        print(f"[INFO] Checkpoint {checkpoint_num} results:")
        print(f"  Mean Return: {summary['mean_return']:.3f} ± {summary['std_return']:.3f}")
        print(f"  Mean Length: {summary['mean_length']:.1f} ± {summary['std_length']:.1f}")
        print(f"  Episodes: {summary['episode_count']}")
        if saved_video_path:
            print(f"  Video: {saved_video_path}")
            
        # except Exception as e:
        #     print(f"[ERROR] Failed to evaluate checkpoint {checkpoint_num}: {e}")
        #     continue
    
    # Log summary table and final results
    if all_results and not args_cli.disable_wandb_logging:
        try:
            # Create summary table 
            summary_table = wandb.Table(columns=["checkpoint", "mean_return", "std_return", "mean_length", "episode_count"])
            for checkpoint_num, results in all_results.items():
                summary_table.add_data(
                    checkpoint_num,
                    results["mean_return"],
                    results["std_return"], 
                    results["mean_length"],
                    results["episode_count"]
                )
            
            # Use the final step for summary
            final_step = max(all_results.keys()) + 1000  # Add offset for summary
            best_checkpoint = max(all_results.keys(), key=lambda k: all_results[k]["mean_return"])
            worst_checkpoint = min(all_results.keys(), key=lambda k: all_results[k]["mean_return"])
            
            final_summary = {
                "summary_table": summary_table,
                # "total_checkpoints_evaluated": len(all_results),
                # "best_checkpoint_number": best_checkpoint,
                # "best_mean_return": all_results[best_checkpoint]["mean_return"],
                # "worst_checkpoint_number": worst_checkpoint, 
                # "worst_mean_return": all_results[worst_checkpoint]["mean_return"],
                # "return_improvement": all_results[best_checkpoint]["mean_return"] - all_results[worst_checkpoint]["mean_return"],
                # "evaluation_complete": True,
            }
            
            wandb.log(final_summary, step=final_step)
            print("[INFO] Successfully logged summary table to separate evaluation run")
            
        except Exception as e:
            print(f"[ERROR] Failed to log summary table: {e}")
        
        print(f"\n🎉 [INFO] Evaluation completed for {len(all_results)} checkpoints")
        print(f"📊 [INFO] Results logged to SEPARATE WandB evaluation run")
        print(f"🔗 [INFO] Evaluation dashboard: {wandb.run.url}")
        print(f"🔗 [INFO] Original training run: {source_run.url}")
        
        # Print all checkpoints and their results
        print(f"\n📈 [INFO] Evaluation Results Summary:")
        for checkpoint_num, results in sorted(all_results.items()):
            print(f"   Checkpoint {checkpoint_num:>5}: Return={results['mean_return']:>6.3f} ± {results['std_return']:.3f}, Length={results['mean_length']:>5.1f}")
        
        best_ckpt = max(all_results.keys(), key=lambda k: all_results[k]["mean_return"])
        print(f"\n🏆 [INFO] Best performing checkpoint: {best_ckpt} (Return: {all_results[best_ckpt]['mean_return']:.3f})")
        
        print(f"\n💡 [INFO] This evaluation run is SEPARATE from training to avoid conflicts")
        print(f"   - Training continues in: {source_run.url}")
        print(f"   - Evaluation results in: {wandb.run.url}")
        print(f"   - Videos and metrics won't be overwritten by ongoing training!")
    
    # Clean up
    env.close()
    
    # Ensure all data is synced before finishing
    if not args_cli.disable_wandb_logging:
        try:
            print("[INFO] Syncing WandB data...")
            if wandb.run is not None:
                wandb.finish()
                print("[INFO] WandB run finished successfully")
            else:
                print("[WARNING] No active WandB run to finish")
        except Exception as e:
            print(f"[WARNING] Error finishing WandB run: {e}")
            try:
                # Force finish if needed
                wandb.finish(exit_code=0)
            except:
                pass


def main():
    """Test environment or play with RSL-RL agent."""
    
    # Check if we're in evaluation mode
    if args_cli.eval_mode:
        run_evaluation_mode(args_cli)
        return
    
    # Original single-checkpoint testing code continues below...
    # parse configuration
    env_cfg = parse_env_cfg(
        args_cli.task, device=args_cli.device, num_envs=args_cli.num_envs, use_fabric=not args_cli.disable_fabric
    )
    env_cfg.recorders = None
    # Attempt to parse RSL-RL config for potential policy loading
    # We need this even if not loading, to check the args
    agent_cfg: RslRlOnPolicyRunnerCfg = cli_args.parse_rsl_rl_cfg(args_cli.task, args_cli)

    # Setup logging directory
    run_timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    log_dir_suffix = "" # Suffix to potentially add experiment info if policy is loaded

    # Apply the csv file if specified
    if args_cli.csv_file is not None:
        if args_cli.csv_file == "LAFAN1_ALL":
            env_cfg.commands.upper_body_joints.demo_start_range = [[args_cli.csv_start, args_cli.csv_start] for _ in env_cfg.commands.upper_body_joints.demo_recording_path]
            env_cfg.commands.upper_body_joints.demo_random_init = False
        else:
            env_cfg.commands.upper_body_joints.demo_recording_path = [args_cli.csv_file]
            env_cfg.commands.upper_body_joints.demo_start_range = [[args_cli.csv_start, args_cli.csv_start]]
            env_cfg.commands.upper_body_joints.demo_random_init = False
            # env_cfg.recorders = None
        if args_cli.csv_end > 0.0:
            env_cfg.commands.upper_body_joints.resampling_time_range = [args_cli.csv_end-args_cli.csv_start, args_cli.csv_end-args_cli.csv_start]

    # input("Press Enter to confirm CSV settings...")
    # env_cfg.commands.upper_body_joints.demo_random_init = False
    # env_cfg.commands.upper_body_joints.demo_zero_reset = True
    # env_cfg.events.reset_robot.params["jpos_range"] = (0.0, 0.0)

    # --- Check if we should load a policy ---
    policy = None
    runner = None
    resume_path = None
    load_policy = False

    # Check for WandB loading first
    if not args_cli.use_zero_action and args_cli.load_run_wandb_id is not None:
        if args_cli.project_entity is None:
            raise ValueError("`--project_entity` must be specified when using `--load_run_wandb_id`.")

        print(f"[INFO] Attempting to load model from WandB run: {args_cli.load_run_wandb_id}")
        import wandb

        api = wandb.Api()
        wandb_project = "isaaclab"  # This can be changed or made into a CLI argument if needed
        run = api.run(f"{args_cli.project_entity}/{wandb_project}/{args_cli.load_run_wandb_id}")

        # --- Update agent config from wandb ---
        # RSL-RL logs the config dictionary directly. We must override local config
        # to ensure the policy architecture matches the loaded checkpoint.
        wandb_cfg = run.config
        
        # Get run name and experiment name for local storage
        run_name = wandb_cfg.get("log_dir", "unknown_run").split("/")[-1]
        experiment_name = wandb_cfg.get("experiment_name", args_cli.task)

        # Create a local directory to store the downloaded model
        log_root_path = os.path.join("logs", "wandb_downloads", experiment_name)
        log_root_path = os.path.abspath(log_root_path)
        checkpoint_dir = os.path.join(log_root_path, run_name)
        os.makedirs(checkpoint_dir, exist_ok=True)
        print(f"[INFO] Downloading model checkpoint to: {checkpoint_dir}")

        # Use the checkpoint name from CLI args, which is consistent with RSL-RL's runner
        checkpoint_pattern = agent_cfg.load_checkpoint
        if not checkpoint_pattern.endswith(".pt"):
            checkpoint_pattern += ".pt"

        # Create patterns for both .pt and .jit files
        jit_pattern = checkpoint_pattern.replace(".pt", ".jit")

        # Convert patterns to proper regex (handle common wildcards)
        pt_regex = checkpoint_pattern.replace(".*", r".*").replace("*", r".*")
        pt_regex = f"^{pt_regex}$"  # Ensure full match

        jit_regex = jit_pattern.replace(".*", r".*").replace("*", r".*")
        jit_regex = f"^{jit_regex}$"  # Ensure full match

        print(f"[INFO] Looking for checkpoint matching pattern: {checkpoint_pattern}")
        print(f"[INFO] Looking for JIT file matching pattern: {jit_pattern}")
        print(f"[INFO] Using PT regex: {pt_regex}")
        print(f"[INFO] Using JIT regex: {jit_regex}")

        # Find files matching both patterns
        matching_pt_files = []
        matching_jit_files = []
        pt_compiled_pattern = re.compile(pt_regex)
        jit_compiled_pattern = re.compile(jit_regex)

        for file in run.files():
            if pt_compiled_pattern.match(file.name):
                matching_pt_files.append(file.name)
            elif jit_compiled_pattern.match(file.name):
                matching_jit_files.append(file.name)

        if not matching_pt_files:
            print(f"[WARNING] No checkpoint files found matching pattern '{checkpoint_pattern}'. Available files:")
            for file in run.files():
                if file.name.endswith('.pt'):  # Only show .pt files for relevance
                    print(f"  - {file.name}")
            print("Will use zero actions.")
        else:
            # Sort and use the latest/best match (you can customize this logic)
            matching_pt_files.sort()
            matching_jit_files.sort()
            
            checkpoint_filename = matching_pt_files[-1]  # Take the last one after sorting
            print(f"[INFO] Found {len(matching_pt_files)} matching PT files, using: {checkpoint_filename}")
            
            # Download the checkpoint file
            run.file(checkpoint_filename).download(root=checkpoint_dir, replace=True)
            resume_path = os.path.join(checkpoint_dir, checkpoint_filename)

            # Also download matching JIT file if available
            jit_filename = None
            jit_path = None
            if matching_jit_files:
                jit_filename = matching_jit_files[-1]  # Take the last matching JIT file
                print(f"[INFO] Found {len(matching_jit_files)} matching JIT files, downloading: {jit_filename}")
                try:
                    run.file(jit_filename).download(root=checkpoint_dir, replace=True)
                    jit_path = os.path.join(checkpoint_dir, jit_filename)
                    if os.path.exists(jit_path):
                        print(f"[INFO] Successfully downloaded JIT file: {jit_path}")
                    else:
                        print(f"[WARNING] JIT file download failed")
                        jit_path = None
                except Exception as e:
                    print(f"[WARNING] Error downloading JIT file: {e}")
                    jit_path = None
            else:
                print(f"[INFO] No matching JIT files found")

            if os.path.exists(resume_path):
                print(f"[INFO] Successfully downloaded checkpoint: {resume_path}")
                load_policy = True
                log_dir_suffix = f"_POLICY_WANDB_{args_cli.load_run_wandb_id}_{checkpoint_filename}"
                
                # Save JIT path for potential later use
                if jit_path:
                    print(f"[INFO] JIT file also available at: {jit_path}")
            else:
                print(f"[WARNING] Failed to download checkpoint. Will use zero actions.")

    # Fallback to local file loading if wandb is not used
    elif not args_cli.use_zero_action and agent_cfg.load_run is not None and agent_cfg.load_checkpoint is not None:
        log_root_path = os.path.join("logs", "rsl_rl", agent_cfg.experiment_name)
        log_root_path = os.path.abspath(log_root_path)
        print(f"[INFO] Attempting to load experiment from directory: {log_root_path}")
        try:
            resume_path = get_checkpoint_path(log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint)
            if os.path.exists(resume_path):
                print(f"[INFO] Found checkpoint: {resume_path}")
                load_policy = True
                policy_log_dir = os.path.dirname(resume_path) # Use policy's log dir for context
                log_dir_suffix = f"_POLICY_{agent_cfg.load_run}_{agent_cfg.load_checkpoint}"
            else:
                 print(f"[WARNING] Checkpoint path not found: {resume_path}. Will use zero actions.")
        except FileNotFoundError:
            print(f"[WARNING] Could not find run directory or checkpoint for run '{agent_cfg.load_run}' with checkpoint '{agent_cfg.load_checkpoint}' in '{log_root_path}'. Will use zero actions.")
        except ValueError as e:
             print(f"[WARNING] Error finding checkpoint: {e}. Will use zero actions.")

    # Setup logging directory *after* determining if policy is loaded
    base_log_dir = os.path.join(
        "logs", "test_runs", args_cli.task, run_timestamp + log_dir_suffix
    )
    os.makedirs(base_log_dir, exist_ok=True)
    print(f"[INFO] Logging results to: {base_log_dir}")

    # Create a subdirectory for the logged data
    data_log_dir = os.path.join(base_log_dir, "step_data")
    os.makedirs(data_log_dir, exist_ok=True)

    # --- Instantiate the logger ---
    # data_logger = DataLogger(log_dir=data_log_dir, filename="simulation_data.csv")
    if hasattr(env_cfg, "recorders") and env_cfg.recorders is not None:
        env_cfg.recorders.dataset_export_dir_path = data_log_dir
        env_cfg.recorders.dataset_filename = "simulation_data.csv"
    # ------------------------------

    if args_cli.video:
        # print(args_cli.video_length, env_cfg.decimation, env_cfg.sim.dt)
        # input()
        time_limit_s = args_cli.video_length * env_cfg.decimation * env_cfg.sim.dt - 0.05
    else:
        # time_limit_s = 1800.0
        # time_limit_s = 300.0
        # time_limit_s = 120.0
        time_limit_s = 30.0
        # time_limit_s = 10.0
        # time_limit_s = 5.0
        # time_limit_s = 1.0

    # env_cfg.episode_length_s = 1.0
    # env_cfg.episode_length_s = time_limit_s
    time_limit = int((time_limit_s + 0.5) / (env_cfg.decimation * env_cfg.sim.dt))

    # Adjust terrain for deterministic playback if necessary
    if hasattr(env_cfg, "scene") and hasattr(env_cfg.scene, "terrain") and env_cfg.scene.terrain is not None:
        if hasattr(env_cfg.scene.terrain, "terrain_generator"):
            terrain_gen_cfg = env_cfg.scene.terrain.terrain_generator
            # Check for existence before setting to avoid errors with different terrain types
            if hasattr(terrain_gen_cfg, 'num_rows'):
                terrain_gen_cfg.num_rows = 5 #1
                print(f"[INFO] Set terrain generator num_rows = {terrain_gen_cfg.num_rows}")
            if hasattr(terrain_gen_cfg, 'num_cols'):
                terrain_gen_cfg.num_cols = 5 #1 #min(args_cli.num_envs // 5 + 1, 10)
                print(f"[INFO] Set terrain generator num_cols = {terrain_gen_cfg.num_cols}")
            if hasattr(terrain_gen_cfg, 'size'):
                terrain_gen_cfg.size = [2.0, 2.0]
            if hasattr(terrain_gen_cfg, 'curriculum'):
                 terrain_gen_cfg.curriculum = False
                 print("[INFO] Disabled terrain curriculum")
            # set flat terrain
            if hasattr(terrain_gen_cfg, 'sub_terrains'):
                terrain_gen_cfg.sub_terrains = {
                    "flat": terrain_gen.MeshPlaneTerrainCfg(proportion=1.0),
                }
        # It might be useful to fix the terrain seed as well if randomization is involved
        # if hasattr(env_cfg.scene.terrain, 'terrain_props') and hasattr(env_cfg.scene.terrain.terrain_props, 'seed'):
        #     env_cfg.scene.terrain.terrain_props.seed = 42 # Example fixed seed

    # create isaac environment
    render_mode = "rgb_array" if args_cli.video else None
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode=render_mode)

    # wrap for video recording
    if args_cli.video:
        # video_kwargs = {
        #     "video_folder": os.path.join(base_log_dir, "videos"), # Save in the main run dir
        #     "name_prefix": f"{args_cli.task}_run",
        #     "step_trigger": lambda step: step == 0,
        #     "video_length": args_cli.video_length,
        #     "disable_logger": True,
        # }
        # print("[INFO] Recording video.")
        # print_dict(video_kwargs, nesting=4)
        # env = gym.wrappers.RecordVideo(env, **video_kwargs)
        import imageio
        os.makedirs(os.path.join(base_log_dir, "videos"), exist_ok=True)
        mp4_writer = imageio.get_writer(os.path.join(base_log_dir, "videos", f"{args_cli.task}_run.mp4"), fps=50)
    else:
        print("[INFO] Video recording disabled.")

    # convert to single-agent instance if required
    is_marl = isinstance(env.unwrapped, DirectMARLEnv)
    if is_marl:
        print("[INFO] Converting Multi-Agent environment to Single-Agent.")
        env = multi_agent_to_single_agent(env)

    # wrap around environment for rsl-rl compatibility
    env = RslRlVecEnvWrapper(env)

    # --- Load Policy if specified ---
    if load_policy:
        print(f"[INFO]: Loading model checkpoint from: {resume_path}")
        # Use a temporary directory for OnPolicyRunner logs, or None
        runner_log_dir = None # os.path.join(base_log_dir, "temp_runner_logs")
        # Create the runner
        import rsl_rl.runners.on_policy_runner
        from softmimic_gym.rsl_rl_extensions.runners import JitSavingOnPolicyRunner as OnPolicyRunner

        runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=runner_log_dir, device=agent_cfg.device)
        # write git state to logs
        # runner.add_git_repo_to_log(__file__)
        # save resume path before creating a new log_dir
        runner.load(resume_path)

        # Obtain the trained policy for inference
        policy = runner.get_inference_policy(device=env.unwrapped.device)

        print("[INFO] Policy loaded successfully.")

        # Export policy to onnx/jit (optional, but good practice)
        export_model_dir = os.path.join(base_log_dir, "exported_policy")
        os.makedirs(export_model_dir, exist_ok=True)
        try:
            print(f"[INFO] Exporting policy to JIT: {os.path.join(export_model_dir, 'policy.jit')}")
            export_policy_as_jit(
                runner.alg.policy, runner.obs_normalizer, path=export_model_dir, filename="policy.jit"
            )
            print(f"[INFO] Exporting policy to ONNX: {os.path.join(export_model_dir, 'policy.onnx')}")
            export_policy_as_onnx(
                runner.alg.policy, normalizer=runner.obs_normalizer, path=export_model_dir, filename="policy.onnx"
            )
        except Exception as e:
             print(f"[ERROR] Failed to export policy: {e}")
    else:
        print("[INFO] No policy loaded. Using zero actions.")
    # ------------------------------

    # reset environment
    print("[INFO] Resetting environment...")
    # env.reset()
    if hasattr(env.env.env, "command_manager"):
        env.env.env.command_manager.compute(dt=env.env.env.step_dt)
    else:
        env.env.env.env.command_manager.compute(dt=env.env.env.env.step_dt)
    obs, info = env.get_observations() # RslRlVecEnvWrapper specific
    timestep = 0
    print("[INFO] Starting simulation loop...")

    mean_l2_tracking_reward = 0.0
    mean_action_l2_reward = 0.0

    # simulate environment
    while simulation_app.is_running():
        # run everything in inference mode
        with torch.inference_mode():
            # --- Log Pre-Step Data ---
            # Use env.env to access the underlying Isaac Lab environment for logger
            # data_logger.log_pre_step(env=env.env, obs=obs, timestep=timestep)

            # Determine actions: Use policy if loaded, otherwise use zeros
            if policy is not None:
                actions = policy(obs)
            else:
                # Ensure zero actions are on the correct device and have correct shape/type
                actions = torch.zeros(env.num_envs, env.num_actions, device=args_cli.device, dtype=torch.float32)
                # actions = torch.randn(env.num_envs, env.num_actions, device=args_cli.device, dtype=torch.float32) * 0.18


            # env stepping using RSLRL wrapper's step
            # The wrapper handles the internal tuple return of Isaac Lab envs
            next_obs_tuple = env.step(actions)
            # Get the *next* observation state for the next loop iteration
            # Note: RSLRL wrapper returns obs slightly differently than raw IsaacLab env
            # next_obs, _ = env.get_observations() # Get next obs *after* step
            obs = next_obs_tuple[0] # Get next obs from the tuple
            info = next_obs_tuple[3]  # Get info from the tuple

            # Increment timestep count AFTER processing the current step
            timestep += 1

            if timestep % 100 == 0:
                print(f"Simulation step: {timestep}")
                print("[INFO] Mean L2 Tracking Reward: ", mean_l2_tracking_reward / 100)
                print("[INFO] Mean Action L2 Reward: ", mean_action_l2_reward / 100)
                mean_l2_tracking_reward = 0.0
                mean_action_l2_reward = 0.0

            # Termination conditions
            if args_cli.video:
                # Exit after recording one video
                if timestep % int(0.02 / (env_cfg.decimation * env_cfg.sim.dt)) == 0:
                    mp4_writer.append_data(env.env.env.render())
                if timestep >= args_cli.video_length:
                    mp4_writer.close()
                    print(f"[INFO] Reached video length ({args_cli.video_length}). Stopping simulation.")
                    break
            else:
                # Default time limit if not recording video
                if timestep >= time_limit:
                   print(f"[INFO] Reached default step limit ({time_limit}). Stopping simulation.")
                   break

    print("[INFO] Simulation loop finished.")

    # --- Save logged data ---
    # data_logger.save()
    print(f"[INFO] Saved simulation data to: {data_log_dir}")
    # ------------------------

    # close the environment
    print("[INFO] Closing environment...")
    env.close()
    print("[INFO] Environment closed.")


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    if simulation_app is not None and simulation_app.is_running():
         print("[INFO] Closing simulation app...")
         simulation_app.close()
         print("[INFO] Simulation app closed.")
    elif simulation_app is not None:
         print("[INFO] Simulation app already closed.")
