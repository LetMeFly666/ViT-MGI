# while true; do
#     python main.py --attackMethod=backdoor --defendMethod=Both --device="cuda:0"
#     python main.py --attackMethod=backdoor --defendMethod=PCA --device="cuda:0"
#     python main.py --attackMethod=lable --defendMethod=Both --device="cuda:0"
#     python main.py --attackMethod=lable --defendMethod=PCA --device="cuda:0"
#     python main.py --attackMethod=grad --defendMethod=Both --device="cuda:0"
#     python main.py --attackMethod=grad --defendMethod=PCA --device="cuda:0"
# done

python main.py --ifFindAttack=False --attackMethod=grad --attackList="[]" --device="cuda:0"
python main.py --ifFindAttack=False --attackMethod=grad --attackList="[0, 1]" --attack_rate=1 --device="cuda:0"
python main.py --ifFindAttack=False --attackMethod=grad --attackList="[0, 1]" --attack_rate=2 --device="cuda:0"
python main.py --ifFindAttack=False --attackMethod=grad --attackList="[0, 1]" --attack_rate=3 --device="cuda:0"
python main.py --ifFindAttack=False --attackMethod=grad --attackList="[0, 1]" --attack_rate=4 --device="cuda:0"