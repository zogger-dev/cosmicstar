import sys
import re

def extract_strings(filepath, min_length=4):
    with open(filepath, 'rb') as f:
        data = f.read()
    
    # Try to find sequences of printable ASCII characters
    strings = []
    current_string = ""
    for byte in data:
        if 32 <= byte <= 126:
            current_string += chr(byte)
        else:
            if len(current_string) >= min_length:
                strings.append(current_string)
            current_string = ""
    
    if len(current_string) >= min_length:
        strings.append(current_string)
        
    print(f"Extracted {len(strings)} strings (min length {min_length}):")
    for i, s in enumerate(strings[:50]): # Print first 50
        print(f"{i}: {s}")
    
    if len(strings) > 50:
         print(f"... and {len(strings) - 50} more")

if __name__ == "__main__":
    if len(sys.argv) > 1:
        extract_strings(sys.argv[1])
    else:
        print("Usage: python analyze_bin.py <file>")
