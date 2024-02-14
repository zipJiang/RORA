__X__ = 0.674
with open('/home/ylu130/workspace/REV-reimpl/log/rev2_t5.txt', 'r') as f:
    results = {}
    for line in f:
        if line.startswith('Experiments for'):
            remove_threshold = float(line.split('removal threshold=')[-1].split(')')[0])
            # initialize dictionary if not exists
            if remove_threshold not in results:
                results[remove_threshold] = {100.0: {'g': None, 'gl': None, 's': None, 'l': None, 'gpt-4_demo=2_raw=True': None, 'gpt-3.5-turbo_demo=2_raw=True': None, 'Llama-2-7b-hf_demo=2_raw=True': None, 'flan-t5-large_demo=2_raw=True':None}, 
                                             10.0: {'g': None, 'gl': None, 's': None, 'l': None, 'gpt-4_demo=2_raw=True': None, 'gpt-3.5-turbo_demo=2_raw=True': None, 'Llama-2-7b-hf_demo=2_raw=True': None, 'flan-t5-large_demo=2_raw=True':None}}
        elif line.startswith('Evaluating rationale format'):
            rationale_format = line.split('Evaluating rationale format: ')[-1].split(' with')[0].strip()
            if rationale_format == 'n':
                continue
            irm_coefficient = float(line.split('with IRM coefficient ')[-1].strip())
        elif line.startswith('{'):
            loss = float(line.split("{'loss': ")[-1].split(',')[0])
            if rationale_format == 'n':
                continue
            results[remove_threshold][irm_coefficient][rationale_format] = round(__X__ - loss, 3)
            

__X__ = 0.691
with open('/home/ylu130/workspace/REV-reimpl/log/rev2_eval.txt', 'r') as f:
    results = {}
    for line in f:
        if line.startswith('Evaluating rationale format'):
            remove_threshold = float(line.split('remove threshold: ')[-1].split('and')[0].strip())
            irm_coef = float(line.split('IRM coefficient: ')[-1].split('using')[0].strip())
            # initialize dictionary if not exists
            used_rationale_format = line.split("using ratioanle format: ")[-1].strip()
            if used_rationale_format not in results:
                results[used_rationale_format] = {0.01: {100.0: {'g': None, 
                                              'gl': None, 
                                              's': None, 
                                              'l': None, 
                                              'gpt-4_demo=2_raw=True': None, 
                                              'gpt-3.5-turbo_demo=2_raw=True': None, 
                                              'Llama-2-7b-hf_demo=2_raw=True': None, 
                                              'flan-t5-large_demo=2_raw=True':None, 
                                              't5-large_demo=0_raw=False': None, 
                                              'gpt2_demo=0_raw=False': None},
                                              10.0: {'g': None, 
                                              'gl': None, 
                                              's': None, 
                                              'l': None, 
                                              'gpt-4_demo=2_raw=True': None, 
                                              'gpt-3.5-turbo_demo=2_raw=True': None, 
                                              'Llama-2-7b-hf_demo=2_raw=True': None, 
                                              'flan-t5-large_demo=2_raw=True':None, 
                                              't5-large_demo=0_raw=False': None, 
                                              'gpt2_demo=0_raw=False': None}
                                        }, 0.005: {100.0: {'g': None, 
                                              'gl': None, 
                                              's': None, 
                                              'l': None, 
                                              'gpt-4_demo=2_raw=True': None, 
                                              'gpt-3.5-turbo_demo=2_raw=True': None, 
                                              'Llama-2-7b-hf_demo=2_raw=True': None, 
                                              'flan-t5-large_demo=2_raw=True':None, 
                                              't5-large_demo=0_raw=False': None, 
                                              'gpt2_demo=0_raw=False': None},
                                              10.0: {'g': None, 
                                              'gl': None, 
                                              's': None, 
                                              'l': None, 
                                              'gpt-4_demo=2_raw=True': None, 
                                              'gpt-3.5-turbo_demo=2_raw=True': None, 
                                              'Llama-2-7b-hf_demo=2_raw=True': None, 
                                              'flan-t5-large_demo=2_raw=True':None, 
                                              't5-large_demo=0_raw=False': None, 
                                              'gpt2_demo=0_raw=False': None}
                                        }}

            rationale_format = line.split('Evaluating rationale format: ')[-1].split(' with')[0].strip()
            if rationale_format == 'n':
                continue
        elif line.startswith('{'):
            loss = float(line.split("{'loss': ")[-1].split(',')[0])
            if rationale_format == 'n':
                continue
            results[used_rationale_format][remove_threshold][irm_coef][rationale_format] = round(__X__ - loss, 3)
        else:
            continue
            