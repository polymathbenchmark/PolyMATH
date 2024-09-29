import json
import base64
import boto3
from dotenv import load_dotenv

from .base_inference_model import BaseInferenceModel

load_dotenv("../../.env")


class BedRockInference(BaseInferenceModel):
    def __init__(self,
                 model_id,
                 max_tokens=512,
                 temperature=0.5,
                 top_p=1,
                 system_prompt="",
                 prompting_strategy="zeroshot"
                 ):
        super().__init__(model_id)
        self.model_id = model_id
        self.prompting_strategy = prompting_strategy
        self.bedrock_client = boto3.client('bedrock-runtime', region_name='us-east-1')
        self.config = {
                "anthropic_version": "bedrock-2023-05-31",
                "max_tokens": max_tokens,
                "temperature": temperature,
                "top_p": top_p,
                "system": system_prompt
            }

    def local_image_to_data_url(self, image_path):
        with open(image_path, "rb") as self:
            image_binary = self.read()
            base_64_encoded_data = base64.b64encode(image_binary)
            base64_string = base_64_encoded_data.decode('utf-8')
        return base64_string

    def get_template(self, prompt, image_path):
        template = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": prompt
                    },
                    {
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": "image/jpeg",
                            "data": self.local_image_to_data_url(image_path)
                        }
                    }
                ],
            }
        ]
        return template

    def get_response(self, prompt, image_path, **kwargs):
        inputs = self.get_template(prompt=prompt, image_path=image_path)
        self.config["messages"] = inputs
        body = json.dumps(self.config)
        response = self.bedrock_client.invoke_model(body=body, modelId=self.model_id)
        response_body = json.loads(response.get('body').read())
        output = response_body['content'][0]['text']
        return output
