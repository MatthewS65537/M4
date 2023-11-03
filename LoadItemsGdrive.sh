mkdir data/ZuCo
# ZuCoTargetStringsEmbeds.pkl (1MVLpaoOprEyz-obuAAYB3Ipo60dro3Gu)
wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=1MVLpaoOprEyz-obuAAYB3Ipo60dro3Gu' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=1MVLpaoOprEyz-obuAAYB3Ipo60dro3Gu" -O data/ZuCoTargetStringsEmbeds.pkl && rm -rf /tmp/cookies.txt
# ZuCoProcessedDatasetDict.pkl (1mfe5kP28yjnrOjMRhQA34X8IRhxtm1MD)
wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=1mfe5kP28yjnrOjMRhQA34X8IRhxtm1MD' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=1mfe5kP28yjnrOjMRhQA34X8IRhxtm1MD" -O data/ZuCo/ZuCoProcessedDatasetDict.pkl && rm -rf /tmp/cookies.txt

mkdir data/Brain2Image
# image_net_dict.pkl (1ZfEXLtOttxcWGKhV_NHCOwOicPCtoLoA)
wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=1ZfEXLtOttxcWGKhV_NHCOwOicPCtoLoA' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=1ZfEXLtOttxcWGKhV_NHCOwOicPCtoLoA" -O data/Brain2Image/image_net_dict.pkl && rm -rf /tmp/cookies.txt
# imageNet_labeled_eeg.pkl (19hLUdO15JKogTbCntKlLeayXk-HpHwE)
wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=19hLUdO15JKogTbCntKlLeayXk-HpHwE-' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=19hLUdO15JKogTbCntKlLeayXk-HpHwE-" -O data/Brain2Image/image_net_dict.pkl && rm -rf /tmp/cookies.txt