


# To convert images_arrays to TFRecords format

def convert_to_tfrecord(images,labels,num_examples,name,directory):
    
    def _int64_feature(value):
        return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))
    
    def _bytes_feature(value):
        return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))
    
    if images.shape[0] != num_examples:
        raise ValueError('Images size %d doesnot match with label size %d',%(images.shape[0],num_examples))
    rows=images.shape[1]
    columns=images.shape[2]
    depth=images.shape[3]
    
    filename=os.path.join(directory,name+'.tfrecords')
    print('Writing ',filename)
    
    writer=tf.python_io.TFRecordWriter(filename)
    
    for index in range len(num_examples):
        image_raw=images(index).tobytes()
        example=tf.train.Example(features=tf.train.Features(feature=
                                                           {
                                                               'height':_int64_feature(rows),
                                                               'width':_int64_feature(columns),
                                                               'depth':_int64_feature(depth),
                                                               'label':_int64_feature(labels(index)),
                                                               'image_raw':_bytes_feature(image_raw)
                                                           }))
        writer.write(example.SerializeToString())
    writer.close()
    