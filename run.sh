python3 main.py --action=train --dataset=gtea --split=1 --arch=dda_ca_enc --version=exp1 --feature=bridge_pretrained --epoch=20
python3 main.py --action=train --dataset=gtea --split=2 --arch=dda_ca_enc --version=exp1 --feature=bridge_pretrained --epoch=20
python3 main.py --action=train --dataset=gtea --split=3 --arch=dda_ca_enc --version=exp1 --feature=bridge_pretrained --epoch=20
python3 main.py --action=train --dataset=gtea --split=4 --arch=dda_ca_enc --version=exp1 --feature=bridge_pretrained --epoch=20

python3 main.py --action=predict --dataset=gtea --split=1 --arch=dda_ca_enc --version=exp1 --feature=bridge_pretrained --epoch=20
python3 main.py --action=predict --dataset=gtea --split=2 --arch=dda_ca_enc --version=exp1 --feature=bridge_pretrained --epoch=20
python3 main.py --action=predict --dataset=gtea --split=3 --arch=dda_ca_enc --version=exp1 --feature=bridge_pretrained --epoch=20
python3 main.py --action=predict --dataset=gtea --split=4 --arch=dda_ca_enc --version=exp1 --feature=bridge_pretrained --epoch=20

python3 eval.py --dataset=gtea --split=0 --version=exp1
python3 eval.py --dataset=gtea --split=0 --version=exp1
python3 eval.py --dataset=gtea --split=0 --version=exp1
python3 eval.py --dataset=gtea --split=0 --version=exp1