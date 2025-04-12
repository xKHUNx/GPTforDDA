# -*- encoding: utf-8 -*-
'''
file       : VLLMManager.py
Description: Integration with vLLM for high-performance LLM inference
Date       : 2024/05/15
Author     : Goh Kun Shun (vibe coded with Trae)
'''

import json
import requests
import time
import os
import threading
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm

class VLLMManager:
    def __init__(self):
        self.lock = threading.Lock()
        
    def generate_response(self, prompt, model="llama2", base_url="http://localhost:8000", temperature=0.7, max_tokens=2048):
        """
        Generate a response from a vLLM-hosted model
        
        Args:
            prompt (str): The input prompt
            model (str): The model name in vLLM
            base_url (str): The vLLM API base URL
            temperature (float): Controls randomness (0.0 to 1.0)
            max_tokens (int): Maximum tokens to generate
            
        Returns:
            str: The generated response
        """
        url = f"{base_url}/generate"
        
        payload = {
            "model": model,
            "prompt": prompt,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "stop": [],  # Optional stop sequences
            "n": 1,      # Number of completions
            "stream": False
        }
        
        try:
            response = requests.post(url, json=payload)
            response.raise_for_status()
            return response.json().get("text", [""])[0]
        except requests.exceptions.RequestException as e:
            print(f"Error calling vLLM API: {e}")
            return ""
    
    def process_example(self, example, model, base_url, temperature, max_tokens):
        """Process a single example and return the result"""
        prompt = example.get("instruction", "")
        response = self.generate_response(
            prompt=prompt,
            model=model,
            base_url=base_url,
            temperature=temperature,
            max_tokens=max_tokens
        )
        
        result = example.copy()
        result["output"] = response
        return result
    
    def save_result(self, result, pool_output_dir):
        """Save a single result to a file in the pool directory"""
        os.makedirs(pool_output_dir, exist_ok=True)
        id_str = result.get("id", "unknown")
        output_file = os.path.join(pool_output_dir, f"{id_str}.json")
        
        with self.lock:
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(result, f, ensure_ascii=False, indent=2)
    
    def merge_results(self, pool_output_dir, output_file_name):
        """Merge all results from the pool directory into a single file"""
        results = []
        for filename in os.listdir(pool_output_dir):
            if filename.endswith('.json'):
                file_path = os.path.join(pool_output_dir, filename)
                with open(file_path, 'r', encoding='utf-8') as f:
                    results.append(json.load(f))
        
        with open(output_file_name, 'w', encoding='utf-8') as f:
            for result in results:
                f.write(json.dumps(result, ensure_ascii=False) + '\n')
    
    def generate_sequences(self, model="llama2", base_url="http://localhost:8000", 
                          input_file_name=None, pool_output_dir=None, 
                          output_file_name=None, processes_num=4,
                          temperature=0.7, max_tokens=2048):
        """
        Generate responses for all examples in the input file
        
        Args:
            model (str): The model name in vLLM
            base_url (str): The vLLM API base URL
            input_file_name (str): Path to input JSONL file
            pool_output_dir (str): Directory to save individual results
            output_file_name (str): Path to save the merged results
            processes_num (int): Number of parallel processes
            temperature (float): Controls randomness (0.0 to 1.0)
            max_tokens (int): Maximum tokens to generate
        """
        # Load examples from input file
        examples = []
        with open(input_file_name, 'r', encoding='utf-8') as f:
            for line in f:
                examples.append(json.loads(line))
        
        # Create pool output directory if it doesn't exist
        os.makedirs(pool_output_dir, exist_ok=True)
        
        # Process examples in parallel
        with ThreadPoolExecutor(max_workers=processes_num) as executor:
            futures = []
            for example in examples:
                future = executor.submit(
                    self.process_example, 
                    example, 
                    model, 
                    base_url, 
                    temperature,
                    max_tokens
                )
                futures.append(future)
            
            # Process results as they complete
            for future in tqdm(futures, desc="Processing examples"):
                result = future.result()
                self.save_result(result, pool_output_dir)
        
        # Merge results into a single file
        self.merge_results(pool_output_dir, output_file_name)
        print(f"All results saved to {output_file_name}")