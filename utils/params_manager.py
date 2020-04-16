
import json

class ParamsManager(object):
    
    def __init__(self, params_file):
        self.params = json.load(open(params_file, 'r'))
        
    def get_params(self):
        return self.params
    
    def get_env_params(self):
        return self.params['env']
    
    def get_agent_params(self):
        return self.params['agent']
    
    def update_agent_params(self, **kwargs):
        for key, value in kwargs.items():
            if key in self.params['agent'].keys():
                self.params['agent'][key] = value
                
    def export_env_params(self, file_name):
        with open(file_name, 'w') as f:
            json.dump(self.params['env'], f, indent=4, separators=(',', ': '), sort_keys=True)
            f.write("\n")
            
    def export_agent_params(self, file_name):
        with open(file_name, 'w') as f:
            json.dump(self.params['env'], f, indent=4, separators=(',', ': '), sort_keys=True)
            f.write("\n")
                
    
if __name__ == "__main__":
    print("Testing ParamsManager...")
    param_file = "parameters.json"
    params_manager = ParamsManager(param_file)
    agent_params = params_manager.get_agent_params()
    print("Agent Params:")
    for k, v in agent_params.items():
        print(k, ":", v)
    env_params = params_manager.get_env_params()
    print("Environment Params:")
    for k, v in env_params.items():
        print(k, ":", v)
    params_manager.update_agent_params(lr=0.01, gamma=0.95)
    updated_agent_params = params_manager.get_agent_params()
    print("Updated Agent Params:")
    for k, v in updated_agent_params.items():
        print(k, ":", v)
        
    print("ParamsManager test completed.")