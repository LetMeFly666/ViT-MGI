while true; do
    python main.py --attackMethod=backdoor --defendMethod=Both --device="cuda:0"
    python main.py --attackMethod=backdoor --defendMethod=PCA --device="cuda:0"
    python main.py --attackMethod=lable --defendMethod=Both --device="cuda:0"
    python main.py --attackMethod=lable --defendMethod=PCA --device="cuda:0"
    python main.py --attackMethod=grad --defendMethod=Both --device="cuda:0"
    python main.py --attackMethod=grad --defendMethod=PCA --device="cuda:0"
done