def runLength(qBlock, DCpred):
    runSymbols = []
    # Differential encoding for DC coefficient
    DC_diff = qBlock[0][0] - DCpred
    runSymbols.append((0, DC_diff))
    # Flatten the 8x8 quantization block into a 1D list using zig-zag scanning
    zigzag = []
    for i in range(8):
        for j in range(8):
            zigzag.append(qBlock[i][j])
    # Run-length encoding for AC coefficients
    count = 0
    for val in zigzag[1:]:  # Exclude DC coefficient (index 0)
        if val == 0:
            count += 1
        else:
            runSymbols.append((count, val))
            count = 0
    # End of block marker
    runSymbols.append((0, 0))
    return runSymbols

def irunLength(runSymbols, DCpred):
    qBlock = [[0 for _ in range(8)] for _ in range(8)]
    # Decoding DC coefficient
    DC_diff = runSymbols[0][1]
    qBlock[0][0] = DC_diff + DCpred
    # Decoding AC coefficients
    zigzag = [0] * 64
    index = 1
    for run, symbol in runSymbols[1:]:
        if symbol == 0:  # End of block marker
            break
        for _ in range(run):
            zigzag[index] = 0
            index += 1
        zigzag[index] = symbol
        index += 1
    # Convert zigzag sequence back to 8x8 block
    row, col = 0, 0
    for i in range(1, 64):
        if col < 7:
            col += 1
        else:
            row += 1
            col = 0
        qBlock[row][col] = zigzag[i]
    return qBlock
