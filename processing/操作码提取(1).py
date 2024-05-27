import pandas as pd

def check(c,i,ops):
    while(i<len(c)):  #len(c)
        if c[i] == '0x':
            i += 1
            continue
        byte = '0x' + c[i]
        # print(byte)
        if byte in czm:
            if byte == '0x60':
                ops.append(mapcode[byte])
                i += 2
                continue
            if byte == '0x61':
                ops.append(mapcode[byte])
                i += 3
                continue
            if byte == '0x62':
                ops.append(mapcode[byte])
                i += 4
                continue
            if byte == '0x63':
                ops.append(mapcode[byte])
                i += 5
                continue
            if byte == '0x64':
                ops.append(mapcode[byte])
                i += 6
                continue
            if byte == '0x65':
                ops.append(mapcode[byte])
                i += 7
                continue
            if byte == '0x66':
                ops.append(mapcode[byte])
                i += 8
                continue
            if byte == '0x67':
                ops.append(mapcode[byte])
                i += 9
                continue
            if byte == '0x68':
                ops.append(mapcode[byte])
                i += 10
                continue
            if byte == '0x69':
                ops.append(mapcode[byte])
                i += 11
                continue
            if byte == '0x6a':
                ops.append(mapcode[byte])
                i += 12
                continue
            if byte == '0x6b':
                ops.append(mapcode[byte])
                i += 13
                continue
            if byte == '0x6c':
                ops.append(mapcode[byte])
                i += 14
                continue
            if byte == '0x6d':
                ops.append(mapcode[byte])
                i += 15
                continue
            if byte == '0x6e':
                ops.append(mapcode[byte])
                i += 16
                continue
            if byte == '0x6f':
                ops.append(mapcode[byte])
                i += 17
                continue
            ops.append(mapcode[byte])
            i += 1
            if byte == '0x70':
                ops.append(mapcode[byte])
                i += 18
                continue
            if byte == '0x71':
                ops.append(mapcode[byte])
                i += 19
                continue
            if byte == '0x72':
                ops.append(mapcode[byte])
                i += 20
                continue
            if byte == '0x73':
                ops.append(mapcode[byte])
                i += 21
                continue
            if byte == '0x74':
                ops.append(mapcode[byte])
                i += 22
                continue
            if byte == '0x75':
                ops.append(mapcode[byte])
                i += 23
                continue
            if byte == '0x76':
                ops.append(mapcode[byte])
                i += 24
                continue
            if byte == '0x77':
                ops.append(mapcode[byte])
                i += 25
                continue
            if byte == '0x78':
                ops.append(mapcode[byte])
                i += 26
                continue
            if byte == '0x79':
                ops.append(mapcode[byte])
                i += 27
                continue
            if byte == '0x7a':
                ops.append(mapcode[byte])
                i += 28
                continue
            if byte == '0x7b':
                ops.append(mapcode[byte])
                i += 29
                continue
            if byte == '0x7c':
                ops.append(mapcode[byte])
                i += 30
                continue
            if byte == '0x7d':
                ops.append(mapcode[byte])
                i += 31
                continue
            if byte == '0x7e':
                ops.append(mapcode[byte])
                i += 32
                continue
            if byte == '0x7f':
                ops.append(mapcode[byte])
                i += 33
                continue
        else:
            #直接跳过没有的操作码

            i += 1
    # return ops

df = pd.read_csv('/big-mult/valid/valid-new-0.csv')
# print(df.head())
df1 = pd.read_csv('../操作码.csv')
df['ops_new'] = 0
len_czm = len(df['bytecode'])

czm = df1['操作码']
czm = list(czm)
hy = df1['含义']

mapcode = {'0x00': 'STOP', '0x01': 'ADD', '0x02': 'MUL', '0x03': 'SUB', '0x04': 'DIV', '0x05': 'SDIV', '0x06': 'MOD', '0x07': 'SMOD', '0x08': 'ADDMOD', '0x0a': 'EXP', '0x0b': 'SIGNEXTEND', '0x10': 'LT', '0x11': 'GT', '0x12': 'SLT', '0x13': 'SGT', '0x14': 'EQ', '0x15': 'ISZERO', '0x16': 'AND', '0x17': 'OR', '0x18': 'XOR', '0x19': 'NOT', '0x1a': 'BYTE', '0x1b': 'SHL', '0x1c': 'SHR', '0x1d': 'SAR', '0x20': 'KECCAK256', '0x30': 'ADDRESS', '0x31': 'BALANCE', '0x32': 'ORIGIN', '0x33': 'CLLLER', '0x34': 'CALLVALUE', '0x35': 'CALLDATALOAD', '0x36': 'CALLDATASIZE', '0x37': 'CALLDATACOPY', '0x38': 'CODESIZE', '0x39': 'CODECOPY', '0x3a': 'GASPRICE', '0x3b': 'EXTCODESIZE', '0x3c': 'EXTCODECOPY', '0x3d': 'RETUTNDATASIZE', '0x3e': 'RETURNDATACOPY', '0x3f': 'EXTCODEHASH', '0x40': 'BLOCKHASH', '0x41': 'COINBASE', '0x42': 'TIMESTAMP', '0x43': 'NUMBER', '0x44': 'DIFFICULTY', '0x45': 'GASLIMIT', '0x46': 'CHAINID', '0x48': 'BASEFEE', '0x50': 'POP', '0x51': 'MLOAD', '0x52': 'MSTORE', '0x53': 'MSTORE8', '0x54': 'SLOAD', '0x55': 'SSTORE', '0x56': 'JUMP', '0x57': 'JUMPI', '0x58': 'GETPC', '0x59': 'MSIZE', '0x5a': 'GAS', '0x5b': 'JUMPDEST', '0x60': 'PUSH', '0x61': 'PUSH', '0x62': 'PUSH', '0x63': 'PUSH', '0x64': 'PUSH', '0x65': 'PUSH', '0x66': 'PUSH', '0x67': 'PUSH', '0x68': 'PUSH', '0x69': 'PUSH', '0x6a': 'PUSH', '0x6b': 'PUSH', '0x6c': 'PUSH', '0x6d': 'PUSH', '0x6e': 'PUSH', '0x6f': 'PUSH', '0x70': 'PUSH', '0x71': 'PUSH', '0x72': 'PUSH', '0x73': 'PUSH', '0x74': 'PUSH', '0x75': 'PUSH', '0x76': 'PUSH', '0x77': 'PUSH', '0x78': 'PUSH', '0x79': 'PUSH', '0x7a': 'PUSH', '0x7b': 'PUSH', '0x7c': 'PUSH', '0x7d': 'PUSH', '0x7e': 'PUSH', '0x7f': 'PUSH', '0x80': 'DUP', '0x81': 'DUP', '0x82': 'DUP', '0x83': 'DUP', '0x84': 'DUP', '0x85': 'DUP', '0x86': 'DUP', '0x87': 'DUP', '0x88': 'DUP', '0x89': 'DUP', '0x8a': 'DUP', '0x8b': 'DUP', '0x8c': 'DUP', '0x8d': 'DUP', '0x8e': 'DUP', '0x8f': 'DUP', '0x90': 'SWAP', '0x91': 'SWAP', '0x92': 'SWAP', '0x93': 'SWAP', '0x94': 'SWAP', '0x95': 'SWAP', '0x96': 'SWAP', '0x97': 'SWAP', '0x98': 'SWAP', '0x99': 'SWAP', '0x9a': 'SWAP', '0x9b': 'SWAP', '0x9c': 'SWAP', '0x9d': 'SWAP', '0x9e': 'SWAP', '0x9f': 'SWAP', '0xa0': 'LOG', '0xa1': 'LOG', '0xa2': 'LOG', '0xa3': 'LOG', '0xa4': 'LOG', '0xb0': 'JUMPTO', '0xb1': 'JUMPIF', '0xb2': 'JUMPSUB', '0xb4': 'JUMPSUBV', '0xb5': 'BEGINSUB', '0xb6': 'BEGINDATA', '0xb8': 'RETURNSUB', '0xb9': 'PUTLOCAL', '0xba': 'GETLOCAL', '0xe1': 'SLOADBYTES', '0xe2': 'SSTOREBYTES', '0xe3': 'SSIZE', '0xf0': 'CREATE', '0xf1': 'CALL', '0xf2': 'CALLCODE', '0xf3': 'RETURN', '0xf4': 'DELEGATECALL', '0xf5': 'CREATE2', '0xfa': 'STATICCALL', '0xfc': 'TXEXECGAS', '0xfd': 'REVERT', '0xfe': 'INVALID', '0xff': 'SELFDESTRUCT'}

ans = 0

for i in range(len_czm):
    code = df.loc[i,'bytecode']
    c = []
    code = code.casefold()
    code = iter(code)
    for x in code:
        c.append(x + next(code))
    ops = []
    check(c, 0, ops)
    print(len(ops))

    if len(ops) < 5000:
        ans += 1
    opsstr = ','.join(map(str, ops))

    df.loc[i, 'ops_new'] = opsstr

df.to_csv('D:/PycharmProjects/pythonProject/Ming21/只能合约/big-mult/valid/valid-new_xiugai0.csv',index=False)
print(ans)
print(df.head())

