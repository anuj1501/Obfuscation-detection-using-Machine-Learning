import math
import sys, os, binascii 

def entropy(input_file):

    with open(input_file,"r",encoding="utf-8") as f:

        javascript_code = f.read()
    
    encoded_string1 = javascript_code.encode()

    byteArr = bytearray(encoded_string1)

    freqList = [0] * len(javascript_code)

    for b in range(256):

        ctr = 0.0
        for byte in byteArr:
            if byte == b:
                ctr += 1  
        freqList.append(float(ctr) / len(javascript_code))

    ent = 0.0
    for freq in freqList:
        if freq > 0:
            ent = ent + (freq * math.log(freq, 2))
    ent = -ent

    return ent, len(javascript_code)