import os
import argparse
import numpy as np

class MultiJob(object):
    def __init__(self, event, start=0, end=125):
        self.event = event
        self.start = start
        self.end = end
        self.total_event = 125
        self.submission_file_name = 'submission_0_125.csv'

    def get_filename(self, index):
        return '_' + "{:03}".format(index) + '_' + str(self.event) 

    def kick_off(self):
        for ii in np.arange(self.start, self.end, self.event):
            nohup_name = 'nohup' + self.get_filename(ii) + '.out'
            print('nohup python hits_clustering.py --test ' + str(ii) + ' ' + str(self.event) + ' > ' + nohup_name + ' 2> /dev/null &')
            os.system('nohup python hits_clustering.py --test ' + str(ii) + ' ' + str(self.event) + ' > ' + nohup_name + ' 2> /dev/null &')

    def merge_submission(self):
        if not os.path.exists(self.submission_file_name):
            print('Create the file ' + self.submission_file_name)
            os.mknod(self.submission_file_name)
        else:
            print('File exists, delete the file ' + self.submission_file_name)
            os.system('rm ' + self.submission_file_name)

        for ii in np.arange(0, self.total_event, self.event):
            submission_name = 'submission' + self.get_filename(ii) + '.csv'
            print('cat ' + submission_name + ' >> ' + self.submission_file_name)
            os.system('cat ' + submission_name + ' >> ' + self.submission_file_name)
            
        
    def compress_submission(self):
        print('gzip ' + self.submission_file_name)
        os.system('gzip ' + self.submission_file_name)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('action', type=str, choices=['run', 'submit'])
    parser.add_argument('--event', nargs=1, type=int)
    parser.add_argument('--start', nargs=1, type=int)
    parser.add_argument('--end', nargs=1, type=int)
    args = parser.parse_args()

    event = 10
    start = 0
    end = 125
    if args.event is not None:
        event = args.event[0]
    if args.start is not None:
        start = args.start[0]
    if args.end is not None:
        end = args.end[0]    

    multiJob = MultiJob(event, start, end)

    if args.action == 'run':
        multiJob.kick_off()
    if args.action == 'submit':
        multiJob.merge_submission()
        multiJob.compress_submission()



    

