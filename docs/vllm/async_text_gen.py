#!/usr/bin/env python3

"""
async_text_gen.py: example python script for text generation with vLLM
"""

import aiohttp
import asyncio
import os
from time import time

vllm_port = os.environ["API_PORT"]

async def post_request(session, url, data):
    try:
        async with session.post(url, json=data) as response:
            if response.status == 200:
                response_data = await response.json()
                choices = response_data.get('choices', [{}])
                text = choices[0].get('text', 'no text')
                return data['prompt'], f"[text] {text}"
            else:
                return data['prompt'], f"[error] {response.status}"
    except asyncio.TimeoutError:
        return data['prompt'], "[error] timeout"


async def main():
    url = f"http://localhost:{vllm_port}/v1/completions"

    # Create a TCPConnector with a limited number of connections
    conn = aiohttp.TCPConnector(limit=512)

    async with aiohttp.ClientSession(connector=conn) as session:
        tasks = [
            post_request(session, url, {
                "model": os.environ['HF_MODEL'],
                "prompt": f"Hint is {idx}, make up some random things."
            })
            for idx in range(10000)
        ]
        responses = await asyncio.gather(*tasks)

    for prompt, response in responses:
        print(f"prompt: {prompt}; response: {response}")


t0 = time()
asyncio.run(main())
print(f"Inference taks {time() - t0} seconds")
