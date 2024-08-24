import torch

CONFIG = {
'generate_config' : {
'temperature' : 0.1,
'top_p' : 0.75,
'top_k' : 40,
'num_beams' : 4,
'max_length' : 356,
#'stream_output' : False,
#'return_dict_in_generate' : True,
#'output_scores' :True
},

'model_config' : {
'load_in_8bit' : True,
'torch_dtype' : torch.float32,
'device_map' : 'auto',
'token' : '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'
},

'token_config' : {
'pad_token_id':(0), 
'bos_token_id':(1), 
'eos_token_id':(2), 
'padding_side':'left', 
'add_eos_token':False
},

'model_List' : {
    'pre_trained_model_list' : ['meta-llama/Meta-Llama-3-8B',
                                'Qwen/Qwen-7B', 
                                '01-ai/Yi-6B-200K', 
                                'mistralai/Mistral-7B-v0.1'],

    'fine_tunded_model_list' : ['meta-llama/Meta-Llama-3-8B', # In Huggingface
                                'Enoch/Llama-7b-hf@tloen/alpaca-lora-7b', # With lora
                                'chat-gpt',  # closed source
                                'MasterAI-EAM-Darwin', # Not in Huggingface
                                ]
}
}

# Generate_config = {
# 'temperature' : 0.1,
# 'top_p' : 0.75,
# 'top_k' : 40,
# 'num_beams' : 4,
# 'max_tokens' : 256,
# 'stream_output' : False,
# 'return_dict_in_generate' : True,
# 'output_scores' :True
# }

# Model_loading_config = {
# 'load_in_8bit' : True,
# 'torch_dtype' : torch.float16,
# 'device_map' : 'auto',
# 'token' : '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'
# }

# token_config = {
# 'pad_token_id':(0), 
# 'bos_token_id':(1), 
# 'eos_token_id':(2), 
# 'padding_side':'left', 
# 'add_eos_token':False
# }

# Model_List = {
#     'Pre_trained_model_list' : ['meta-llama/Llama-2-7b-hf', 
#                                 'Qwen/Qwen-7B', 
#                                 '01-ai/Yi-6B-200K', 
#                                 'mistralai/Mistral-7B-v0.1'],

#     'Fine_tunded_model_list' : ['meta-llama/Llama-2-7b-chat-hf', # In Huggingface
#                                 'Enoch/Llama-7b-hf@+@tloen/alpaca-lora-7b', # With lora
#                                 'chat-gpt',  # closed source
#                                 'MasterAI-EAM-Darwin', # Not in Huggingface
#                                 ]
# }