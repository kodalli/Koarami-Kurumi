import random
import re
from llm.inference.deciLM_7b import DeciLM7b
import typing as t
import pandas as pd

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

def generate_prompts(system_prompts: list[str]) -> list[str]:
    '''
    Given a list of base system prompts,
    this function generates a list of variants on these prompts using generate_variants_for
    '''
    # NOTE(TG): If we don't choose a singular base prompt *before* generating variants,
    # certain base prompts can have a lot more appearances in the final list to choose from
    # due to the amount of variants.
    unflattened_list = [list(generate_variants_for(x)) for x in system_prompts]

    # flattened_list: list[str] = []
    # for l in unflattened_list:
    #     flattened_list += l

    return unflattened_list

# The regex used to find message variants (e.g.: `%{Hi|Hello} there!`)
VARIANT_REGEX = re.compile(r'%{(.+?)}')


def generate_variants_for(
        string: str,
        max_generations: int | None = 20000,
        start_counter_at: int = 0) -> t.Generator[str, None, None]:
    '''
    Given a string like "%{Hello|Hi} there%{.|!}, this should yield:

    - Hello there.
    - Hello there!
    - Hi there.
    - Hi there!
    '''

    # Some bot creators went wild with the variants, which causes ridiculous
    # generations if we try to exhaust all possibilities so we cap that here.
    # `start_counter_at` is used for keeping track across recursive calls.
    counter = start_counter_at

    if (match := re.search(VARIANT_REGEX, string)) is not None:
        # Once we have a "%{X|Y|Z}" matched inside the original string, we:
        # - Fetch .groups()[0] (which will give us `X|Y|Z`)
        # - Split by `|` (so we have ["X", "Y", "Z"])
        # - Filter out empty strings
        alternatives = filter(lambda x: x.strip(), match.groups()[0].split("|"))

        # Then, we break the string apart into what comes before and after the
        # alternatives, that way we can re-build with "prefix + choice + sufix".
        prefix = string[:match.start()]
        sufix = string[match.end():]

        for alternative in alternatives:
            variant = f'{prefix}{alternative}{sufix}'

            # However, some strings have multiple variant blocks. In that case,
            # we operate on them recursively until we have just regular strings
            # after generating all possible variants.
            still_have_match = re.search(VARIANT_REGEX, variant) is not None
            if still_have_match:
                for inner_variant in generate_variants_for(
                        variant, start_counter_at=counter):
                    yield inner_variant

                    # Keep track and break after `max_generations`.
                    counter += 1
                    if max_generations is not None and counter >= max_generations:
                        break
            else:
                yield variant

                # Keep track and break after `max_generations`.
                counter += 1
                if max_generations is not None and counter >= max_generations:
                    break
    else:
        yield string

def create_sys_info(source, label_bot="Koarami Kurumi", label_user="Twitch Chat", limit_data_length=4096, ensure_final_bot_message=False):
    # tokenizer.sep_token = '<|sep|>'
    # tokenizer.pad_token = '<|pad|>'
    # tokenizer.cls_token = '<|cls|>'
    # tokenizer.mask_token = '<|mask|>'

    # tokens_header = len(tokenizer(sysinfo)['input_ids'])          
    # # Drop late messages. Can help making summaries more accurate for
    # # the start of the conversation
    # i_start = 0                
    # for i_end in range(len(source['conversation']), i_start, -1):
    #     totaltokens = tokens_header + sum(tokens_messages[:i_end])
    #     if totaltokens < limit_data_length:
    #         break
    
    # if ensure_final_bot_message:
    #     # When dropping late messages it's possible that the conversation
    #     # can end with a user response. If this happens, drop that message
    #     # so that the final message is certainly a bot response (<FIRST>).
    #     if source['conversation'][i_end-1]['name'] == '<SECOND>':
    #         i_end = i_end - 1                

    # if totaltokens > limit_data_length:
    #     # Drop the messages if they exceed the threshold after limiting.
    #     # This can happen if the limiting length is too low.
    #     print(f'[{num+1}]:DROP', end=' ')
    #     # continue
    sysinfo_flipped = [
        "%{Provided|Given|With|Considering|Taking into consideration|Taking into account|Following}"
        " the %{previous|above|prior|preceding} %{conversation|discussion|exchange|chat},"
        " %{personalities|character personas|character information|character attributes} and %{scenario|summary|background}"
        " %{can be|may be} %{inferred|deduced|understood to be|taken to be} %{as follows|like this|as shown below|as shown next}."
        ]

    sysinfo = []

    sysinfo.append(f"{label_bot}'s Persona: {source['persona']['<STREAMER>']}\n")
    sysinfo.append(f"{label_user}'s Description: {source['persona']['<VIEWER>']}\n")
    sysinfo.append(f"Scenario: {source['scenario']}\n")
            
    preamble_a = random.choice([
        f'Take the role of an AI twitch streamer {label_bot}.',
        f'You are {label_bot}, an AI twitch streamer.',
        f'Play the role of {label_bot}, an online AI streamer.',
        f'Write as if you were {label_bot}, an online streamer.',
    ])

    preamble_b = random.choice([                                
        'Taking the above information into consideration,',
        'After carefully considering the above information,',
        'Following the persona and scenario described above,',
        'With personality and character now described,',
    ])

    preamble_c = random.choice([           
        f'you must engage in an evil, banter filled conversation with {label_user} below this line.',
        f'you must make fun of with {label_user} further below.',
        f'you must chat in a smug playful manner with {label_user}.',
    ])

    preamble_d = random.choice([ 
        f"Do not write {label_user}'s dialogue lines in your responses.",
        f"Never write for {label_user} in your responses.",
        f"Do not write chat dialogues for {label_user}.",
        f"Only write chat dialogues for {label_bot}.",
    ])
    
    sysinfo.append(f'{preamble_a} {preamble_b} {preamble_c} {preamble_d}\n\n')
    
    
    sysinfo = '\n'.join(sysinfo)
    # sysinfo = substitute_participants(sysinfo)
    sysinfo = fix_punctuation(sysinfo)

    sysinfo_flipped = [random.choice(generate_prompts(sysinfo_flipped)[0])]
    sysinfo_flipped.append(f"\n{label_bot}'s Persona: {source['persona']['<FIRST>']}\n")
    sysinfo_flipped.append(f"{label_user}'s Persona: {source['persona']['<SECOND>']}\n")
    sysinfo_flipped.append(f"Scenario: {source['scenario']}\n")
    sysinfo_flipped = '\n'.join(sysinfo_flipped)
    # sysinfo_flipped = substitute_participants(sysinfo_flipped)
    sysinfo_flipped = fix_punctuation(sysinfo_flipped)
    return {'role': 'system', 'content': sysinfo}

def prepare_prompt(combined_transcription_file, streamer_name, description_file):
    df = pd.read_csv(combined_transcription_file)
    # Select rows with streamer_name
    streamer_df = df[df['user_name'] == streamer_name]

def apply_template(chat, chat_template, tokenize=False, add_generation_prompt=False):
    conversation_text = DeciLM7b.get_tokenizer().apply_chat_template(chat, chat_template=chat_template, tokenize=tokenize, add_generation_prompt=add_generation_prompt)

def read_description_file(path):
    with open(path, 'r') as f:
        text = f.read()
        # tokenized_text = DeciLM7b.get_tokenizer().encode(text, return_tensors="pt")
        # print(tokenized_text.shape)
        return text

if __name__ == '__main__':
    # out = create_sys_info({"persona": {"<STREAMER>": "An AI streamer named Koarami Kurumi", "<VIEWER>": "Viewers of the Twitch Stream"}, "scenario": "A streamer interacts with Twitch Chat with playful banter."})
    # print(out)
    # read_description_file("data/Toma/description.txt")

    # Instruction, Persona, Scenario
    # Response
    # Input
    # Start with Response and end with Input

    pass