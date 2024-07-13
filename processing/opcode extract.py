import solcx
from pyevmasm import disassemble_hex
import pandas as pd
import re
def replace_opcodes(opcode):
    # Use regular expressions to replace PUSH1 through PUSH32 with PUSH
    opcode = re.sub(r'\bPUSH([1-9]|[1-2][0-9]|3[0-2])\b', 'PUSH', opcode)
    # Use regular expressions to replace SWAP1 through SWAP32 with SWAP
    opcode = re.sub(r'\bSWAP([1-9]|1[0-6])\b', 'SWAP', opcode)
    # Use regular expressions to replace DUP1 through DUP32 with DUP
    opcode = re.sub(r'\bDUP([1-9]|1[0-6])\b', 'DUP', opcode)
    return opcode


if __name__ == '__main__':
    df = pd.read_csv('train.csv') #The dataset is downloaded after it is completed。Change the train.csv to your file address
    len_df = len(df['source_code'])

    for i in range(len_df):
        opcodes_cleaned = []
        bytecode = df.loc[i,'bytecode']
        opcodes = disassemble_hex(bytecode)
        opcodes = opcodes.split()
        opcodes_replaced = [replace_opcodes(opcode) for opcode in opcodes]
        # Use regular expressions to remove all strings that start with 0x
        for j in range(len(opcodes_replaced)):
            if '0x' in opcodes_replaced[j]:
                continue
            else:
                opcodes_cleaned.append(opcodes_replaced[j])
        opcodes_end = ','.join(opcodes_cleaned)
        df.loc[i,'ops'] = opcodes_end
        print(i)
    df.to_csv('train_bytecode.csv',index=False)  #The address of the file you want to store

