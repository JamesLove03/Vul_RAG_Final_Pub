#Add the following line to the gitignore
#common/api-keys/
#also run this command
#git rm --cached common/api-keys/*
#keys to look for specifically

import pickle
from pathlib import Path

fakekey = "fake"
key_file = 'api_keys'

#write all of the keys
with open(f"{key_file}/openkey_openai_api_key.pkl", "wb") as f:
    pickle.dump(fakekey, f)

with open(f"{key_file}/deepseek_api_key.pkl", "wb") as f:
    pickle.dump(fakekey, f)

with open(f"{key_file}/qwen_api_key.pkl", "wb") as f:
    pickle.dump(fakekey, f)

with open(f"{key_file}/claude_api_key.pkl", "wb") as f:
    pickle.dump(fakekey, f)

with open(f"{key_file}/gemini_api_key.pkl", "wb") as f:
    pickle.dump(fakekey, f)


#read all the keys to make sure
with open(f"{key_file}/openkey_openai_api_key.pkl", "rb") as f:
    print(pickle.load(f))
with open(f"{key_file}/gemini_api_key.pkl", "rb") as f:
    print(pickle.load(f))
with open(f"{key_file}/claude_api_key.pkl", "rb") as f:
    print(pickle.load(f))