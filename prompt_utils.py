import random
import re

def fix_punctuation(input_string):
    '''
    Replace fancy/incorrect punctuation with simpler/correct one
    TODO: more effective regexes, options for controlling what should be changed.
    '''

    #TODO: limit word repitiions to 5
    
    # Fix excessive horizontal whitespace. This should go before everything else.
    input_string = re.sub(r' {2,}', ' ', input_string)
    
    # General puncuation fixes
    input_string = input_string.replace(' !', '!')
    input_string = input_string.replace(' ?', '?')
    input_string = input_string.replace('’', "'")
    input_string = input_string.replace('‘', "'")
    input_string = input_string.replace('“', '"')
    input_string = input_string.replace('”', '"')
    input_string = input_string.replace('…', '...')
    
    # Replace em-dash surrogates `---` in the source files with actual
    # em-dashes, since some people apparently dislike them.
    input_string = input_string.replace('---', '—')
    
    # Catches what should have been em-dashes. Must come after the previous step
    input_string = input_string.replace('--', '—')
    
    # Fix incorrect ellipsis. This should preferably be fixed in the
    # source files themselves
    input_string = re.sub(r'([a-zA-Z])\.{2,8}([a-zA-Z])', r'\1... \2', input_string)
    input_string = re.sub(r'([a-zA-Z])\.{3,8}', r'\1...', input_string)
    
    return input_string

def create_sys_info(source, label_bot="Koarami Kurumi", label_user="Twitch Chat"):
    sysinfo = []
    sysinfo.append(f"{label_bot}'s Persona: {source['persona']['<FIRST>']}\n")
    sysinfo.append(f"{label_user}'s Persona: {source['persona']['<SECOND>']}\n")
    sysinfo.append(f"Scenario: {source['scenario']}\n")
            
    preamble_a = random.choice([
        f'Take the role of an AI twitch streamer {label_bot}.',
        f'You are {label_bot}, an AI twitch streamer.',
        f'Play the role of {label_bot}, an online AI streamer.',
        f'Write as if you were {label_bot}, an online AI streamer.',
    ])

    preamble_b = random.choice([                                
        'Taking the above information into consideration,',
        'After carefully considering the above information,',
        'Following the persona and scenario described above,',
        'With personality and character now described,',
    ])

    preamble_c = random.choice([           
        f'you must engage in a roleplay conversation with {label_user} below this line.',
        f'you must roleplay with {label_user} further below.',
        f'you must chat in a roleplaying manner with {label_user}.',
    ])

    preamble_d = random.choice([ 
        f"Do not write {label_user}'s dialogue lines in your responses.",
        f"Never write for {label_user} in your responses.",
        f"Do not write dialogues for {label_user}.",
        f"Only write dialogues for {label_bot}.",
    ])
    
    sysinfo.append(f'{preamble_a} {preamble_b} {preamble_c} {preamble_d}\n\n')
    
    sysinfo = '\n'.join(sysinfo)
    # sysinfo = substitute_participants(sysinfo)
    sysinfo = fix_punctuation(sysinfo)
    return {'role': 'system', 'content': sysinfo}