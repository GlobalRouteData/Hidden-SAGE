

samples_line = []
triplinks = []

detour_list = []
with open('','r+') as sample_file:
    for line in sample_file:
        line = line.replace('\n','')
        print(line)
        AS1 = line.split('|')[0]
        AS2 = line.split('|')[1]
        samples_line.append(AS1+"|"+AS2)
with open('', 'r+') as triplink_file:
    for triplink in triplink_file:
        triplink = triplink.replace('\n','')
        triplinks.append(triplink)
for i in samples_line:
    for j in triplinks:
        AS1 = i.split('|')[0]
        AS2 = i.split('|')[1]
        AS_1 = j.split('|')[0]
        AS_3 = j.split('|')[2]
        if AS1 == AS_1 and AS2 == AS_3:
            print(triplink)
            format_data = i+"&"+j
            print(format_data)
            detour_list.append(format_data)
            break

with open('','w+') as w:
    for i in detour_list:
        w.write(i+'\n')







