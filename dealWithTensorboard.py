import tensorflow as tf
from tensorflow.core.util.event_pb2 import Event

def manipulateTensorbard(pathFile):
    for rec in tf.data.TFRecordDataset([str(pathFile)]):
        ev = Event()
        ev.MergeFromString(rec.numpy())
        print('oi')

def main():
    manipulateTensorbard('runs_workingcopy/Apr01_09-00-21_joaocardia-GA-78LMT-S2/events.out.tfevents.1617278425.joaocardia-GA-78LMT-S2.23947.0')

if __name__ == '__main__':
    main()