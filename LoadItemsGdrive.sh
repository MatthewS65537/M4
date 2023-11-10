mkdir data/ZuCo
mkdir data/ZuCo/task1-SR
mkdir data/ZuCo/task2-NR
mkdir data/ZuCo/task2-NR-2.0
mkdir data/ZuCo/task3-TSR
mkdir data/ZuCo/stanfordsentiment

# ZuCoTargetStringsEmbeds-CLIP.pkl (1MVLpaoOprEyz-obuAAYB3Ipo60dro3Gu)
wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=1MVLpaoOprEyz-obuAAYB3Ipo60dro3Gu' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=1MVLpaoOprEyz-obuAAYB3Ipo60dro3Gu" -O data/ZuCo/ZuCoTargetStringsEmbeds-CLIP.pkl && rm -rf /tmp/cookies.txt
# ZuCoProcessedDatasetDict-CLIP.pkl (1mfe5kP28yjnrOjMRhQA34X8IRhxtm1MD)
wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=1mfe5kP28yjnrOjMRhQA34X8IRhxtm1MD' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=1mfe5kP28yjnrOjMRhQA34X8IRhxtm1MD" -O data/ZuCo/ZuCoProcessedDatasetDict-CLIP.pkl && rm -rf /tmp/cookies.txt
# ZuCoTargetStringsEmbeds-BART.pkl (1iDGa8mCvkWx7EouE4NQwvmkkGBUIaAWC)
wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=1iDGa8mCvkWx7EouE4NQwvmkkGBUIaAWC' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=1iDGa8mCvkWx7EouE4NQwvmkkGBUIaAWC" -O data/ZuCo/ZuCoTargetStringsEmbeds-BART.pkl && rm -rf /tmp/cookies.txt
# ZuCoProcessedDatasetDict-BART.pkl (1_L30anMBHwW2kxNs0J3mdr5HhTtak_M_)
wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=1_L30anMBHwW2kxNs0J3mdr5HhTtak_M_' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=1_L30anMBHwW2kxNs0J3mdr5HhTtak_M_" -O data/ZuCo/ZuCoProcessedDatasetDict-BART.pkl && rm -rf /tmp/cookies.txt

# sentiment_labels.json
wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=1kBCgWqjDPxbpU1e3zb0KL6ijkajzsd2G' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=1kBCgWqjDPxbpU1e3zb0KL6ijkajzsd2G" -O data/ZuCo/task1-SR/sentiment_labels.json && rm -rf /tmp/cookies.txt
# tenary_dataset.json
wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=1TiskkzjSM_zbDMJenf3CUCWOZ-Y4wnTK' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=1TiskkzjSM_zbDMJenf3CUCWOZ-Y4wnTK" -O data/ZuCo/stanfordsentiment/tenary_dataset.json && rm -rf /tmp/cookies.txt

mkdir data/Brain2Image
# image_net_dict.pkl (1ZfEXLtOttxcWGKhV_NHCOwOicPCtoLoA)
wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=1ZfEXLtOttxcWGKhV_NHCOwOicPCtoLoA' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=1ZfEXLtOttxcWGKhV_NHCOwOicPCtoLoA" -O data/Brain2Image/image_net_dict.pkl && rm -rf /tmp/cookies.txt
# imageNet_labeled_eeg.pkl (19hLUdO15JKogTbCntKlLeayXk-HpHwE)
wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=19hLUdO15JKogTbCntKlLeayXk-HpHwE-' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=19hLUdO15JKogTbCntKlLeayXk-HpHwE-" -O data/Brain2Image/imageNet_labeled_eeg.pkl && rm -rf /tmp/cookies.txt