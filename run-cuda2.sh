python main-notPretrained.py --epoch_client=3 --datasize_perclient=320 --learning_rate=0.001 --num_rounds=32 --device="cuda:1"
python main-notPretrained.py --epoch_client=2 --datasize_perclient=320 --learning_rate=0.001 --num_rounds=32 --device="cuda:1"
python main-notPretrained.py --epoch_client=1 --datasize_perclient=320 --learning_rate=0.001 --num_rounds=32 --device="cuda:1"
python main-notPretrained.py --epoch_client=3 --datasize_perclient=320 --batch_size=16 --learning_rate=0.001 --num_rounds=32 --device="cuda:1"
python main-notPretrained.py --epoch_client=3 --datasize_perclient=320 --batch_size=8 --learning_rate=0.001 --num_rounds=32 --device="cuda:1"
python main-notPretrained.py --epoch_client=2 --datasize_perclient=320 --batch_size=8 --learning_rate=0.001 --num_rounds=32 --device="cuda:1"
python main-notPretrained.py --epoch_client=1 --datasize_perclient=320 --batch_size=8 --learning_rate=0.001 --num_rounds=32 --device="cuda:1"

