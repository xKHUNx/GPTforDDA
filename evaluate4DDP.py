
'''
file       :evaluate4DDP.py
Description: GPT for DDP
Date       :2024/03/05 21:17:16
Author     :Yaxin Fan
Email      : yxfansuda@stu.suda.edu.cn
'''



from GPT import BotManager
from GPT.OllamaManager import OllamaManager
from GPT.VLLMManager import VLLMManager
from DDP.Parser import Parser
from DDP.utils import DataAnalysis

if __name__ == '__main__':
    # Choose which LLM provider to use
    use_vllm = False  # Set to True to use vLLM
    use_ollama = False  # Set to True to use Ollama
    
    if use_vllm:
        # using vLLM to generate the response
        bot_manager = VLLMManager()
        processes_num = 8  # Adjust based on your server capacity
        model = "llama2"  # or any other model available in your vLLM server
        base_url = "http://localhost:8000"  # Default vLLM URL
        input_file_name = './DDP/dataset/Molweni/Molweni_instructions.jsonl'
        pool_output_dir = './DDP/ResponseFile/MolweniPool/'
        output_file_name = './DDP/ResponseFile/MolweniResponse.json'
        
        bot_manager.generate_sequences(
            model=model,
            base_url=base_url,
            input_file_name=input_file_name,
            pool_output_dir=pool_output_dir,
            output_file_name=output_file_name,
            processes_num=processes_num,
            temperature=0.7,
            max_tokens=2048
        )
    elif use_ollama:
        # using Ollama to generate the response
        bot_manager = OllamaManager()
        processes_num = 4  # Fewer processes for local inference
        model = "llama3.2"  # or any other model available in your Ollama
        base_url = "http://localhost:11434"  # Default Ollama URL
        input_file_name = './DDP/dataset/Molweni/Molweni_instructions.jsonl'
        pool_output_dir = './DDP/ResponseFile/MolweniPool/'
        output_file_name = './DDP/ResponseFile/MolweniResponse.json'
        
        bot_manager.generate_sequences(
            model=model,
            base_url=base_url,
            input_file_name=input_file_name,
            pool_output_dir=pool_output_dir,
            output_file_name=output_file_name,
            processes_num=processes_num,
            temperature=0.7,
            max_tokens=2048
        )
    else:
        # using GPT to generate the response 
        bot_manager = BotManager()
        processes_num = 50  # num of threads
        api_key = 'sk-xxx'
        base_url = 'https://api.openai.com/v1'
        model = 'gpt-3.5-turbo'
        input_file_name = './DDP/dataset/Molweni/Molweni_instructions.jsonl'
        pool_output_dir = './DDP/ResponseFile/MolweniPool/'
        output_file_name = './DDP/ResponseFile/MolweniResponse.json'
        
        bot_manager.generate_sequences(
            api_key=api_key,
            base_url=base_url,
            model=model,
            input_file_name=input_file_name,
            pool_output_dir=pool_output_dir,
            output_file_name=output_file_name,
            processes_num=processes_num
        )

    # evaluation
    structure_output_file = "./DDP/StructureDir/MolweniStructure.json"
    parser = Parser(output_file_name, structure_output_file)
    parser.write_structure()
    data_analysis = DataAnalysis(structure_output_file, '')
    data_analysis.compute_f1_all_file()
    data_analysis.compute_the_Accuracy_Of_Different_RelaType()
