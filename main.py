import json, sys, os
import torch, pytorch_lightning as pl
import pytorch_lightning.callbacks as cb
from pytorch_lightning.loggers import TensorBoardLogger
from hsqc_split_dataset import HsqcDataModule
from hsqc_split_Unet import UNet



if len(sys.argv) > 1:
        name = sys.argv[1]
else: 
        name = "none"
        

f = open('/root/HSQC_split/configs/'+ name + '.json')
config = json.load(f)

save_dir = '/root/HSQC_split/exp_results'
model = UNet(config)
data_module = HsqcDataModule(batch_size=config['batch_size'])
checkpoint_callback = cb.ModelCheckpoint(monitor="val/mean_loss", mode="min", save_last=True)



tbl = TensorBoardLogger(save_dir=save_dir, name = name, version="v")
trainer = pl.Trainer(max_epochs=config.get('epoch', 200), gpus=torch.cuda.device_count(), logger=tbl, 
                     callbacks=[checkpoint_callback],
                     strategy="ddp_find_unused_parameters_false",
                     auto_scale_batch_size = True)
# print(model.lightning_optimizers())
trainer.fit(model, data_module)#,  ckpt_path=f"/root/tessellation_project/exp_results/{name}/v/checkpoints/last.ckpt")

'''testing stage '''
# ckpts= list(sorted(os.listdir(f"/root/tessellation_project/exp_results/{name}/v/checkpoints")))
# print(ckpts)
# best_ckpt = (f"/root/tessellation_project/exp_results/{name}/v/checkpoints/"+ckpts[-2])
# trainer.test(model, data_module, ckpt_path=best_ckpt)

# best_ckpt = (f"/root/tessellation_project/exp_results/{name}/v/checkpoints/"+ckpts[-1])
# trainer.test(model, data_module, ckpt_path=best_ckpt)
    