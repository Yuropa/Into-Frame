from path_utils import add_project_paths
add_project_paths()

from pathlib import Path
from typing import Any

import numpy as np

from sam2.build_sam import build_sam2_video_predictor_hf
from remote_connection.remote_server import RemoteServer

# Single fixed object id — the client hands us one reference mask per call.
_TRACKED_OBJ_ID = 1


class VideoSegServer(RemoteServer):
    def setup(self):
        self.predictor = build_sam2_video_predictor_hf(
            "facebook/sam2.1-hiera-large",
            device=self.device,
        )

    def perform(self, action: str, temp_path: Path, input: Any) -> Any:
        if action == "segment_video":
            return self._segment_video(
                video_path=input["video_path"],
                reference_mask=input["mask"],
                reference_frame_idx=input.get("reference_frame_idx", 0),
            )
        raise ValueError(f"Unknown action: {action}")

    def _segment_video(self, video_path: str, reference_mask: np.ndarray, reference_frame_idx: int) -> dict:
        self.report_progress(0.0, "Loading video…")
        state = self.predictor.init_state(video_path=video_path)
        num_frames = state["num_frames"]

        self.predictor.add_new_mask(
            state,
            frame_idx=reference_frame_idx,
            obj_id=_TRACKED_OBJ_ID,
            mask=reference_mask,
        )

        masks_by_frame: dict[int, np.ndarray] = {}

        def _consume(frame_gen, direction: str):
            for frame_idx, obj_ids, video_res_masks in frame_gen:
                obj_idx = obj_ids.index(_TRACKED_OBJ_ID)
                masks_by_frame[frame_idx] = (video_res_masks[obj_idx, 0] > 0.0).cpu().numpy()
                self.report_progress(
                    len(masks_by_frame) / num_frames,
                    f"Propagating mask ({direction}) — frame {frame_idx + 1}/{num_frames}…",
                )

        # Forward from the reference frame covers it through the end of the video;
        # only walk backward too if the reference isn't already the first frame,
        # so the whole video ends up with a mask either way.
        _consume(
            self.predictor.propagate_in_video(state, start_frame_idx=reference_frame_idx, reverse=False),
            "forward",
        )
        if reference_frame_idx > 0:
            _consume(
                self.predictor.propagate_in_video(state, start_frame_idx=reference_frame_idx, reverse=True),
                "backward",
            )

        self.predictor.reset_state(state)

        stacked = np.stack([masks_by_frame[i] for i in range(num_frames)], axis=0)
        self.report_progress(1.0, "Done")
        return {"masks": stacked}


if __name__ == "__main__":
    VideoSegServer.run()
