import sys
import os
import glob
import PyPDF2

RANDOM_ID = str(sys.argv[1])
PROCESS = str(sys.argv[2])
FILE_LIST = str(sys.argv[3])

def merge_pdf():
    merger = PyPDF2.PdfMerger()
    file_location = os.path.join('C:/xampp/htdocs/Tools/uploads',RANDOM_ID)
    try:
        for file_name in FILE_LIST.split(','):
            file_path = os.path.join(file_location,file_name)
            merger.append(file_path)
    except:
        print('NULL')

    file_name = file_location +'.pdf'
    merger.write(file_name)
    merger.close()
    print(RANDOM_ID)

if PROCESS == 'merge_pdf':
    merge_pdf()
else:
    print('NULL')