#!/usr/bin/env python3
#
#   vandercorput.py
#
#   Provides the van der Corput sequence, uniformly sampling the 0...1
#   range, that resolves to at least that given fraction.  For
#   example:
#
#      sequence(0.3) = [0.5 0.25 0.75]
#
#   as we need 4 pieces (actual resolution 1/4) to test to a
#   resolution of 0.3.
#
#   Note the classic van der Corput sequence assumes a 1/2^n
#   resolution, giving a perfectly regular sequence.  But this may
#   include more elements than truly necessary.  So the general
#   sequence only adjust the resolution to provide uniform samples:
#
#      power2sequence(0.22) = [1/2 1/4 3/4 1/8 5/8 3/8 7/8]   1/8 resolution
#      sequence(0.22)       = [3/5 2/5 4/5 1/5]               1/5 resolution
#
#   To use:
#
#      import vandercorput
#
#          for delta in vandercorput.sequence(fraction):
#              ....
#
import math


#
#   Classic van der Corput Sequence
#
#   This assumes a 1/2^n resolution (1/4, 1/8, 1/16, ...)
#
def power2sequence(fraction):
    # Determine the number of sections.
    N = math.ceil(-math.log2(fraction))
    if (N < 1):
        return []

    # Initialize the first section.
    delta   = 0.5
    section = [0.5]
    list    = section

    # Keep adding section as needed.
    for _ in range(1, N):
        delta   /= 2.0
        section = [x-delta for x in section] + [x+delta for x in section]
        list = list + section

    # Return the full list.
    return list


#
#   General Sequence
#
def sequence(fraction):
    # Recursively resort a list: first the middle element, then
    # alternating element from the top and bottom halves, where each
    # half has been individually resorted.
    def resort(list):
        k = int(len(list)/2)
        if k > 0:
            top = resort(list[:k])
            bot = resort(list[k+1:])
            list[0] = list[k]
            list[1::2] = top
            list[2::2] = bot
        return list

    # Return the resorted list of 1/n ... (n-1)/n
    n = math.ceil(1.0/fraction)
    return resort([i/n for i in range(1,n)])



#
#   Test
#
if __name__== "__main__":
    print('1.3:', power2sequence(1.3))
    print('0.7:', power2sequence(0.7))
    print('0.4:', power2sequence(0.4))
    print('0.3:', power2sequence(0.3))
    print('0.2:', power2sequence(0.24))
    print('0.1:', power2sequence(0.1))

    print('1.3:', sequence(1.3))
    print('0.7:', sequence(0.7))
    print('0.4:', sequence(0.4))
    print('0.3:', sequence(0.3))
    print('0.2:', sequence(0.24))
    print('0.1:', sequence(0.1))
    print('0.0625:', sequence(0.0625))

    
    
