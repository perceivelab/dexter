"""Generate captions and textual reports from saved frames using VLM/LLM calls."""

import base64
import json
import os
from pathlib import Path
from typing import Dict, List, Optional

from openai import OpenAI
from tqdm import tqdm


class Reasoner:
    """Wraps caption and report generation over saved diffusion frames."""

    def __init__(
        self,
        vlm_id: Optional[str] = None,
        llm_id: Optional[str] = None,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        system_prompt_config_path: Optional[str] = None,
        classifier: Optional[str] = None,
        class_to_activate: Optional[int] = None,
        class_path: Optional[str] = None,
    ):
        self.client = OpenAI(api_key=api_key, base_url=base_url)
        self.classifier = classifier
        self.class_to_activate = class_to_activate
        self.class_path = class_path
        self.vlm_id = vlm_id
        self.llm_id = llm_id
        self.system_prompt_config_path = system_prompt_config_path

        self.system_prompts = self._load_system_prompts()
        self.captions: List[str] = []
        self.report: Optional[str] = None

    def _load_system_prompts(self) -> Dict[str, str]:
        """Load prompt templates from text files if provided."""
        if not self.system_prompt_config_path:
            return {}

        prompts: Dict[str, str] = {}
        try:
            for prompt_file in os.listdir(self.system_prompt_config_path):
                if not prompt_file.endswith(".txt"):
                    continue
                key = prompt_file.split(".")[0]
                with open(
                    os.path.join(self.system_prompt_config_path, prompt_file), "r"
                ) as file_handle:
                    prompts[key] = file_handle.read()
        except Exception as exc:  # keep behavior but fail softly
            print(f"Error loading system prompts: {exc}. Will be using empty prompts.")
        return prompts

    def _chat_completion(self, history: List[Dict], model: Optional[str]) -> str:
        """Small helper to centralize the chat.completions call."""
        response = self.client.chat.completions.create(
            model=model,
            messages=history,
            temperature=0.2,
        )
        return response.choices[0].message.content

    def _encode_image(self, image_path: str) -> str:
        """Read an image and return a base64 string for model consumption."""
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode("utf-8")

    def get_caption(self, image_path: str) -> str:
        """Request a caption for a single image using the configured VLM.

        Args:
            image_path: Path to the PNG image to caption.

        Returns:
            Model-produced caption string.
        """
        base64_image = self._encode_image(image_path)
        history = [
            {"role": "system", "content": self.system_prompts.get("captions_sp", "")},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Describe this picture"},
                    {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{base64_image}"}},
                ],
            },
        ]
        return self._chat_completion(history, model=self.vlm_id)

    def evaluate_caption_from_latents(self) -> List[str]:
        """Loop over generated frames, requesting captions and logging them.

        Returns:
            List of captions in frame order (empty if no frames found).
        """
        img_folder = os.path.join(self.class_path, "frames") if self.class_path else None
        if not img_folder:
            return []

        captions: List[str] = []
        caption_log_path = os.path.join(self.class_path, "caption.txt")
        os.makedirs(self.class_path, exist_ok=True)

        with open(caption_log_path, "a") as log_file:
            for img_name in tqdm(sorted(os.listdir(img_folder)), desc="Generating Captions"):
                img_path = os.path.join(img_folder, img_name)
                caption = self.get_caption(img_path)
                log_file.write(f"{caption}\n----------------------\n")
                captions.append(caption)

        self.captions = captions
        return captions

    def _label_to_text(self, class_idx: int) -> str:
        """Map a class index to its human-readable label."""
        if self.classifier == "celeba_erm":
            labels = {0: "Not Blonde", 1: "Blonde"}
        elif self.classifier == "waterbirds_erm":
            labels = {0: "Landbird", 1: "Waterbird"}
        elif self.classifier in ("fairfaces_resnet18_gender", "fairfaces_resnet18_race"):
            labels = {0: "20 to 29 years old people", 1: "50 to 59 years old people"}
        else:
            labels_path = Path(__file__).resolve().parent.parent / "data" / "imagenet_classes.json"
            with open(labels_path, "r") as file_handle:
                labels = json.load(file_handle)

        return labels[class_idx]

    def get_report(self) -> str:
        """Summarize caption set into a short report with the configured LLM.

        Returns:
            Generated report string, also written to report.txt when possible.
        """
        if self.class_to_activate is None:
            raise ValueError("class_to_activate must be set to generate a report.")

        text_label = self._label_to_text(self.class_to_activate)
        history = [
            {"role": "system", "content": self.system_prompts.get("report_sp", "")},
            {"role": "user", "content": f"Class: {text_label} - Captions: {self.captions}"},
        ]

        report = self._chat_completion(history, model=self.llm_id)
        self.report = report

        if self.class_path:
            with open(os.path.join(self.class_path, "report.txt"), "w") as report_file:
                report_file.write(report)

        return report

    def reason(self) -> (List[str], Optional[str]):
        """End-to-end run: caption all frames then generate a summary report.

        Returns:
            (captions, report) tuple; report may be None if generation fails.
        """
        captions = self.evaluate_caption_from_latents()
        report = self.get_report()
        return captions, report


if __name__ == "__main__":
    #test
    class_folder = "out/292_20251128-170559-babd0823"
    reasoner = Reasoner(
        vlm_id="llava:13b",
        llm_id="gpt-oss:120b",
        api_key="your_api_key",
        base_url="http://localhost:11434/v1",
        system_prompt_config_path="sp_config",
        class_path=class_folder,
        classifier="robust50",
        class_to_activate=292,
    )
    captions, report = reasoner.reason()
