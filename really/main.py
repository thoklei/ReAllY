from hyperparameters import *
from toytrain.load import return_data
from model import SeqToSeq_cnn
from train import train_loop
import tensorflow as tf


if __name__=="__main__":

    tf.keras.backend.clear_session()


    # 5 tage historical cmip - 5 tage historil era 5

    #x = tf.zeros((2,5,15,15,3))

    #t = tf.ones((2,3,15,15,3))

    #ata = [(x,t)]*100
    #print(cell(x, states))
    data = return_data(path='./toytrain/toy_test/',split=5)
    for x,t in data:
            break;
    model = SeqToSeq_cnn(x.shape[2], x.shape[3], x.shape[4])
    #data = return_data()
    train_loop(model,data)

    tf.keras.models.save_model(
    model, './models/', overwrite=True, include_optimizer=True, save_format=None,
    signatures=None, options=None, save_traces=True
    )



    #out = model(data)
    #print(out)
