import torch
import torch.nn as nn

class BaseModel(nn.Module):
    def load_model(self, deblur_path, reblur_path):
        # load deblur model
        if deblur_path != None: 
            print('load_deblur_path: ' + deblur_path)
            self.deblur_model.model.load_state_dict({k.replace('module.', ''):v for k, v in torch.load(deblur_path).items()})

        # load reblur model
        if reblur_path != None: 
            print('load_reblur_path: ' + reblur_path)
            self.reblur_model.model.load_state_dict({k.replace('module.', ''):v for k, v in torch.load(reblur_path).items()})
        
        return 0
           
    def save_model(self, epoch, skip_deblur=False):
        # save reblur model
        tmp = self.reblur_model.model.state_dict()
        model_state_dict = {key.replace('module.',''): tmp[key] for key in tmp}
        model_name = self.args.save_dir[:-1] if self.args.save_dir.endswith('/') else self.args.save_dir+'/model/reblur_'+str(epoch).zfill(5)+'.pt'
        torch.save(model_state_dict, model_name)

        if not skip_deblur:
            # save deblur model
            tmp = self.deblur_model.model.state_dict()
            model_state_dict = {key.replace('module.',''): tmp[key] for key in tmp}
            model_name = self.args.save_dir[:-1] if self.args.save_dir.endswith('/') else self.args.save_dir+'/model/deblur_'+str(epoch).zfill(5)+'.pt'
            torch.save(model_state_dict, model_name)

