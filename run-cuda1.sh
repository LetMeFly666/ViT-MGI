while true; do
    python main.py --attackMethod=backdoor --defendMethod=Both --device="cuda:1"
    python main.py --attackMethod=backdoor --defendMethod=PCA --device="cuda:1"
    python main.py --attackMethod=lable --defendMethod=Both --device="cuda:1"
    python main.py --attackMethod=lable --defendMethod=PCA --device="cuda:1"
    python main.py --attackMethod=grad --defendMethod=Both --device="cuda:1"
    python main.py --attackMethod=grad --defendMethod=PCA --device="cuda:1"
done