from django.shortcuts import render
from django.http import HttpResponse
import os
from django.core.files.base import ContentFile
from django.core.files.storage import default_storage
import tensorflow as tf
import os
import numpy as np
import re
from PIL import Image
import matplotlib.pyplot as plt



class NodeLookup(object):
            def __init__(self, label_lookup_path=None, uid_lookup_path=None):
                if not label_lookup_path:
                    label_lookup_path = os.path.join(
                        model_dir, 'imagenet_2012_challenge_label_map_proto.pbtxt')
                if not uid_lookup_path:
                    uid_lookup_path = os.path.join(
                        model_dir, 'imagenet_synset_to_human_label_map.txt')
                self.node_lookup = self.load(label_lookup_path, uid_lookup_path)
            def load(self, label_lookup_path, uid_lookup_path):
                if not tf.gfile.Exists(uid_lookup_path):
                    tf.logging.fatal('File does not exist %s', uid_lookup_path)
                if not tf.gfile.Exists(label_lookup_path):
                    tf.logging.fatal('File does not exist %s', label_lookup_path)
                proto_as_ascii_lines = tf.gfile.GFile(uid_lookup_path).readlines()
                uid_to_human = {}
                for line in proto_as_ascii_lines:
                    line = line.strip('\n')
                    parse_items = line.split('\t')
                    uid = parse_items[0]
                    human_string = parse_items[1]
                    uid_to_human[uid] = human_string
                proto_as_ascii = tf.gfile.GFile(label_lookup_path).readlines()
                node_id_to_uid = {}
                for line in proto_as_ascii:
                    if line.startswith('  target_class:'):
                        target_class = int(line.split(': ')[1])
                    if line.startswith('  target_class_string:'):
                        target_class_string = line.split(': ')[1]

                        node_id_to_uid[target_class] = target_class_string[1:-2]
                node_id_to_name = {}
                for key, val in node_id_to_uid.items():
                    if val not in uid_to_human:
                        tf.logging.fatal('Failed to locate: %s', val)
                    name = uid_to_human[val]
                    node_id_to_name[key] = name
                return node_id_to_name
            def id_to_string(self, node_id):
                if node_id not in self.node_lookup:
                    return ''
                return self.node_lookup[node_id]



model_dir = 'inception_model'
def updateinfo(request):
    dat=0
    if request.method =="POST":
        stu_photos = request.FILES.get('photo')
        user_name =  request.FILES.get('photo').name
        path = default_storage.save('static/user/photo/'+user_name,ContentFile(stu_photos.read()))
        #image = path
        image = path
        #def create_graph():

        with tf.gfile.FastGFile(os.path.join(
                    model_dir, 'classify_image_graph_def.pb'), 'rb') as f:
                graph_def = tf.GraphDef()
                graph_def.ParseFromString(f.read())
                tf.import_graph_def(graph_def, name='')
        image_data = tf.gfile.FastGFile(image, 'rb').read()
            #create_graph()

        with tf.Session() as sess:

            softmax_tensor = sess.graph.get_tensor_by_name('softmax:0')
            predictions = sess.run(softmax_tensor, {'DecodeJpeg/contents:0': image_data})
            predictions = np.squeeze(predictions)
            node_lookup = NodeLookup()
            top_5 = predictions.argsort()[-5:][::-1]
            top_5_max = top_5.max()
            human_string = node_lookup.id_to_string(top_5_max)
            score = predictions[top_5_max]
            global dat
            dat = '%s (score = %.5f)' % (human_string, score)

                # for node_id in top_5:
                #     human_string = node_lookup.id_to_string(node_id)
                #     score = predictions[node_id]
                #     data = ('%s (score = %.5f)' % (human_string, score))
    #data = {'sssss'}
        #return HttpResponse("上传成功")



    return render(request, 'sddd.html',{'data':dat})



