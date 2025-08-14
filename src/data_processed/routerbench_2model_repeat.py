
from src.data_processed import utils
import pandas as pd
import torch
from tqdm import tqdm
import numpy as np
import os

data_path = "./data/rawdata/routerbench/routerbench_0shot.pkl"
save_path = "./data/processed/all-mpnet-base-v2/routerbench_0shot_2model_rep2"
model_path = "./models/all-mpnet-base-v2"
cost_scale = 1000

# Load raw data
rawdata = pd.read_pickle(data_path)
# print(rawdata.keys())
prompts = rawdata["prompt"].tolist()
print(f"Number of prompts: {len(prompts)}")
available_models_description = {
    # "gpt-3.5-turbo-1106": "A lightweight and cost-efficient model ideal for everyday tasks, generating concise responses with moderate reasoning capabilities.", 
    # "claude-instant-v1": "An affordable model providing quick, straightforward answers best suited for simple queries and short-text generation tasks.",
    # "claude-v1": "A moderately-priced model with enhanced reasoning skills, balancing depth and efficiency for extended conversations.",
    # "claude-v2": "A powerful premium model excelling in complex analysis and long-form content creation with strong contextual understanding.",
    "gpt-4-1106-preview": "A top-tier expensive model delivering exceptional reasoning and creative capabilities for advanced problem-solving.",
    # "meta/llama-2-70b-chat": "A robust open-source model offering strong general-purpose performance at no cost, great for diverse conversational needs.",
    # "mistralai/mixtral-8x7b-chat": "A highly efficient open-source model specialized in multilingual tasks and technical discussions with balanced output length.",
    # "zero-one-ai/Yi-34B-Chat": "A capable bilingual model freely handling both English and Chinese content with mid-length analytical responses.",
    "WizardLM/WizardLM-13B-V1.2": "A free specialized model optimized for complex instruction-following and detailed multi-turn dialogues.",
    # "meta/code-llama-instruct-34b-chat": "A purpose-built coding model freely providing detailed technical explanations and extended code solutions.",
    # "mistralai/mistral-7b-chat": "A compact open-source model delivering fast, focused responses perfect for lightweight applications."
}

# Convert texts to embeddings
deivce = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {deivce}")

prompts_embeddings = utils.text2embeddings(
    prompts,
    model_path=model_path,
    batch_size=32,
    device=deivce
)
print(prompts_embeddings.shape)
modeldesc_embeddings = utils.text2embeddings(
    list(available_models_description.values()),
    model_path=model_path,
    batch_size=1,
    device=deivce
)
available_models_description_embeddings = {
    model: emb.numpy() for model, emb in zip(available_models_description.keys(), modeldesc_embeddings)
}

# get avialable models and write the model description
# print(rawdata_gb_sampleid.get_group("arc-challenge.val.136"))


####################### MD5 #######################
import hashlib

def get_md5(text):
    return hashlib.md5(text.encode('utf-8')).hexdigest()

md5_list = [
    '3f4d460766f087657a3f240eaf663526',
    '17d6faed786393cd52d78506063a0da2',
    'a5c36b187e8266534dd7500ccf0cb0c3',
    'a96a8751caeffcda7d223ad0800b691f',
    '2fcfb2257d4727208bcbe307a4d0d110',
    'bcebf08871a0493f4911584f8896252e',
    '8ecc3f57b37810d575bc2606d794d721',
    '1e9032f3773aeb3bf0ddd5e0e660a0c9',
    'ac2f26f09fe13d389b533c3a80492c15',
    '63ba548b36b6f2638958728462af4fb3',
    '78cf61a48dc61a3feab48b4230223878',
    'd9380bd60d2b5cc0fa9ae2b4ac6c1694',
    '462fc7d5e372b00130934003a29b3aea',
    'a3be9b1fcf1db99872716f488d9d9ec3',
    '4a05014c37ecaa36505013f353be19e1',
    '4e504fe5b5ffdb9b3e002c993fbda422',
    'c12706c4434e3f1a7f194c24900a7f87',
    'ea436e5575e283ae9298a206bb07caa9',
    '90b0abe54565aea0d765063fd0287377',
    '4e5446a81f5ecfa0d358ea42e7d6df07',
    '6ab469c38971092ad28a04b1d03bac9f',
    '3d2163a17ade44d5c3052622bd8f7fab',
    '67079bc20e603204e104a3a93fa15d8a',
    '230510e9c74b54dc402199b7bd80650d',
    'f8afa54e0631bfa86dd147f54af7e890',
    'b40620448f3c150ea800591431a05fca',
    '0f6b232976eca4ceb3a86571d5837f58',
    '8b65d070c977bd3483eff7e420380e3d',
    '64def229548184736cb27a9351c91134',
    '59092831c6e40e9c38467ccb4b20e1e1',
    'a2b2e0d12d2521a15c5054956cac7713',
    '2f5a707611228db193077b7048aa9b40',
    '1062b64ed735e3c7759703c7accdfc24',
    '2be182583a1ecdbaef72685192632918',
    '862d6a953ec14b2146083ca1203e40c7',
    'a53d11f677c59adb57486ac963fc74cb',
    'e5c6012c1f84c7f34edd593a38a1f0d3',
    '59ef59b79656851f617ce631e3684958',
    'ce9e204d64e1e88c801100fdfb53f1ae',
    '5c05d15536e499e5b4b4ec13df61bd88',
    '947f76d417d9106cb203ea47dacc7d27',
    '4a1cd1ce3edbe4df07778d4b24a6f3ed',
    '1258989844c2721ac0c9284ec320d632',
    '161b4ed0a201d517adb4a34664c76821',
    '2f930f883e296ef4a8dff2834985ff65',
    'bab99076be4f73ac73b9d9bf66084485',
    '4a323464900cff6741efb0cd79080dd9',
    '4fbf15f19f7e0289a55d8a12509d2152',
    '1c47fc32b103593e4a3b795b96335f59',
    'a8e2b78574cecc9017c535b76adb6adb',
    '79cb6aa8c0d953e6d5c35324968418e5',
    '6fa09a76db6615d919c8d96cfaf114d0',
    '0891958369961020f1fd9d8f027b922d',
    '3e5204f0c92a94d3d14db307d4e1206e',
    '5fcf536e2f835605ca8f88639d750d1f',
    '49fa323800f323a0bccb6e99d97f63ad',
    '3a27b2bfc73c4b71805b567eb248d520',
    '8eaa6cda95c41edf60e814e252e14685',
    'c228fa44060f56c87bf0030981be15d9',
    '862f73cf93957340a9ef82b2d17708c6',
    'fb407c6f5a17c7132fe0706382fa6e95',
    'ee9fe507addd96e4262861b2e3f93b3e',
    '90c82f54497fb5af967d09ae03329b44',
    '9f6d55f45f188dcb60033ad58b72cbf1',
    '6433e4a068172d8159c00f5a52c7bafe',
    '1b6f0fac39ad530263847a8e7c4c9052',
    '3cac597d655e8962c20d3f12f163a3a2',
    'a1cdb969911ef29105ac1c480c35d0ad',
    '9399ce622071b74f5fd7e1455be36b20',
    '0a5dad8ed9384b9eefe4a7804062b34f',
    '3dfa4de2d6c4f0d51f07d198c30f0882',
    '349f448fff130b9f79018154ec1c0b03',
    '8598f968288c4225f0283fc8e0e033cf',
    'be2ee5e73991fa70647567424ab2e0f2',
    '59ed4879b4c4adb00404180a10e2e002',
    '677b266d450926a1ab3c37f4c09727d1',
    'a7deab93e0a08f89d4dd672ecc07351c',
    'e64f7bd92b4459f6596c6c0265140797',
    '9b7e156c1592e5ed05b94c4543422c7b',
    '5ac06f485dd0f6f735929f2a0d67ad12',
    '7274d79e65786578292ba2d4f5ec5d4f',
    '5d6276895512371e9dfd15d0b52496f3',
    '46b390e96b27f96ae49076b7be73d9f0',
    '125ea6a4a1fb5818b9b23f6a498c25f6',
    '654bc7e24573568cfa06763b0c9dba9c',
    '8ec0f400b22d6aeb583edbda2b0e8d4d',
    '47b34026912a3941c08f1660f37d458f',
    'fd778833831638038969fde47de6489d',
    '059e2f5c12d8036e587909e141c8d798',
    '892b7725756c96670c0582b9bca7deec',
    '055c862d36f4de8104f1dec1afa1fca9',
    'e5d7c424101c3ada07992ae68694c4a7',
    '2333a4d1b1f0e981345c2980e81a9fe2',
    'bce96ffbf83b1454682ee56e41a30426',
    '11bba85d758dbedabdbb922f18fea684',
    'bb81e6ef986952abd84f0c597c7deaba',
    '6b57b2a6346493f02a03e4da5e3bcf35',
    'efe39faa73ca5c9a46049447db9e0987',
    '1402e082d2a66672e4ccf58b200644c7',
    '7ddb73e24807cff8ae8761131383808f',
    'fabe81df4c95aa11568c82c45f532b63',
    '56e6b833b4b8d9930e9ac0fa3bf40e8d',
    'f8be699186697c2a3af0f402989f499e',
    '282597cd043755e611e24ad085bf4cde',
    'fc39be7d672476e01cca089a8bf64bb6',
    'd1333a2da7951540512c23e3e2df011c',
    'a0d1432efe2ffcc5c13c0c3d6130a089',
    'fc593c9423143450c34b8bd59b0f22e3',
    '6719a7513e91a3c507f3c2dee4e1ebc7',
    'b0bf898d7905378f83afa15157b27401',
    '97120a119d6dfe6fcb7afe4c2e4cae27',
    '0489b796ff39ebd4785641242841afb8',
    '1594740fee1177f37e0a1a474a242a3e',
    '24db5f104bbbf56c2f5a8dc080239f00',
    'fa4ef9306c7a5acfc20361bfbbffab65',
    'a87b0894fd63f632bd2e49ee56842f79',
    '2858cdea19c4fc298887126700d40cbe',
    '5858eaf4bc16f253794076215f188169',
    'c106af0b51b93ef79d9f63b55406e69e',
    '65c98957127d166d225ee14afb12f032',
    '141a5b8ee596bf21c84d9e23bf499a51',
    '2a3a7cc8b1c2970ef2435428ac2c9edf',
    'e1c64a50bf23dc1936271310cdfd8c4e',
    'e8ecb652329854d4488dc14763a4c647',
    '07b56e0681a070fbe6b1ddf7c2454ad0',
    '15e2ee85f166d9e6b09b5fd4e3bcb6c0',
    '31dcfc1d14d66330d9eb90cdc2eb3e52',
    '74c1cb4e803e3052d6c4eb4d62d36c42',
    '06eae7352f678ad0f80651e457987cd7',
    'e429c1d245ad171d9ab10a17506f461f',
    'ba9d817a037ea03bf19530e2c0ccc10c',
    'fb16960b045e3975939816fd0a0114e9',
    'fa4ff0920896f80f3af75f5d18babd44',
    'c36d2273e648feafba7c0f0c2b784c2b',
    'b6794549a2a834c7196b2870a50d0093',
    'efeb85f5a44741b248edb69e7976415c',
    '2c037899e43d590ac7152f7e4627a20d',
    '8d690bc53e518b2954a42c89f9c907c9',
    'd1348a07f7fdacd4a77f8add6b991056',
    '44cbc76335e2d831c4d00329ddbfbb80',
    '9e648ef65532a928dd017cf0936fc175',
    '408a022e29cdc5676e009c3d1aee8aa8',
    'a89c9782259054ed692f3c4f40e446b7',
    'e5e0a4f553c7244824879d3ed31cac15',
    '43d4fffada044ec01eb14c01ed68f54e',
    '3144f6c1f92026da0a1dd539d346b2e4',
    'e316cf5bb3c1e68c54dc7e4143d984d0',
    '204abf5b6684df3c8a6552e34c804fb2',
    '3746aaa1d211a7d4b3613827cbcd7213',
    'fbe4e711827ef74c8683e5f20f7ca6cb',
    '2e1ba469befbf53768b6a63edbde001c',
    'c8efb727c5d616b7f04cfbccc18017a7',
    'afa2166d41e159ab7ced3195abb0bcdd',
    '95f94497be7ed4ec978c22b8d1a690ab',
    'fe7daa678f4a78c6bd5e6a049b1d36ea',
    '5c7dd7a408f035290e8210ea2e59238d',
    'dbf258261352db69926d750114334dea',
    'ed52102d97d2ac5ef918079bd3cbb23b',
    '2bcc0049e7f36c7bb05142c87b972cc9',
    '741b2f2ce5ff8d39dc1ad148aa21e3bb',
]

####################### MD5 #######################

repeat_time = 100

processed_dataset = []
for index in tqdm(range(len(rawdata)), desc="Processing data"):
    df = rawdata.iloc[index]
    sample_id = df["sample_id"]
    sample_id = "routerbench_0shot." + sample_id
    prompt = df["prompt"]
    if get_md5(prompt) not in md5_list:
        continue
    gt = {}
    for model in available_models_description.keys():
        gt_per_model = {
            "response": df[model + "|model_response"],
            "reward": df[model],
            "cost": df[model + "|total_cost"] * cost_scale,
        }
        gt[model] = gt_per_model
    processed_data = {
        "sample_id": sample_id,
        "available_models_description": available_models_description,
        "available_models_description_embeddings": available_models_description_embeddings,
        "prompt": prompt,
        "prompt_embedding": prompts_embeddings[prompts.index(prompt)].numpy(),
        "gt": gt,
    }
    # processed_dataset.append(processed_data)
    processed_dataset += [processed_data] * repeat_time

# exit(0)

# randomly split processed_data into train, test
np.random.seed(42)
np.random.shuffle(processed_dataset)
train_size = int(0.8 * len(processed_dataset))
train_dataset = processed_dataset[:train_size]
test_dataset = processed_dataset[train_size:]

# Save processed data as .parquet files
if not os.path.exists(save_path):
    os.makedirs(save_path)


train_df = pd.DataFrame(train_dataset)
train_df.to_parquet(os.path.join(save_path, "online.parquet" if cost_scale == 1.0 else f"online_cost_scale_{cost_scale}.parquet"), engine='pyarrow', index=False)
test_df = pd.DataFrame(test_dataset)
test_df.to_parquet(os.path.join(save_path, "offline.parquet" if cost_scale == 1.0 else f"offline_cost_scale_{cost_scale}.parquet"), engine='pyarrow', index=False)
print(f"Processed data saved to {save_path}")