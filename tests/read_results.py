__X__ = 0.691
with open('/home/ylu130/workspace/REV-reimpl/log/rev2.txt', 'r') as f:
    results = {}
    for line in f:
        if line.startswith('Experiments for'):
            remove_threshold = float(line.split('removal threshold=')[-1].split(')')[0])
            # initialize dictionary
            results[remove_threshold] = {100.0: {'g': None, 'gl': None, 's': None, 'l': None}, 
                                         10.0: {'g': None, 'gl': None, 's': None, 'l': None}}
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
            
