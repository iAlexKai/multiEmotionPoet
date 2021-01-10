with open('train_data.txt') as file, open('out.txt', 'w') as out:
    lines = file.read().strip().split('\n')
    for line in lines:
        content = line.split('\t')
        title = content[0]
        if len(title) > 12:
            continue
        invalid = False
        
        for word in line:
            if ((word > '\u9fa5' or word < '\u4e00') and word != '\t' and word != '，' and word != '。') or word == '□':
                # import pdb
                # pdb.set_trace()
                invalid = True
        if invalid:
            continue
        out.write(line + '\n')