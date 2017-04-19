""" Numerous data generator configurations not natively supported by keras

"""
from keras.preprocessing.image import Iterator, ImageDataGenerator, NumpyArrayIterator, array_to_img, transform_matrix_offset_center, apply_transform, random_channel_shift, flip_axis
import os
import numpy as np

class XOnlyImageDataGenerator(ImageDataGenerator):

    def flow(self, X, batch_size=32, shuffle=True, seed=None,
             save_to_dir=None, save_prefix='', save_format='jpeg'):
        return XOnlyNumpyArrayIterator(
            X, self,
            batch_size=batch_size, shuffle=shuffle, seed=seed,
            dim_ordering=self.dim_ordering,
            save_to_dir=save_to_dir, save_prefix=save_prefix, save_format=save_format)



class XOnlyNumpyArrayIterator(Iterator):

    def __init__(self, X, image_data_generator,
                 batch_size=32, shuffle=False, seed=None,
                 dim_ordering='default',
                 save_to_dir=None, save_prefix='', save_format='jpeg'):
        if dim_ordering == 'default':
            dim_ordering = K.image_dim_ordering()
        self.X = np.asarray(X)
        if self.X.ndim != 4:
            raise ValueError('Input data in `NumpyArrayIterator` '
                             'should have rank 4. You passed an array '
                             'with shape', self.X.shape)
        channels_axis = 3 if dim_ordering == 'tf' else 1
        if self.X.shape[channels_axis] not in {1, 3, 4}:
            raise ValueError('NumpyArrayIterator is set to use the '
                             'dimension ordering convention "' + dim_ordering + '" '
                             '(channels on axis ' + str(channels_axis) + '), i.e. expected '
                             'either 1, 3 or 4 channels on axis ' + str(channels_axis) + '. '
                             'However, it was passed an array with shape ' + str(self.X.shape) +
                             ' (' + str(self.X.shape[channels_axis]) + ' channels).')
        self.image_data_generator = image_data_generator
        self.dim_ordering = dim_ordering
        self.save_to_dir = save_to_dir
        self.save_prefix = save_prefix
        self.save_format = save_format
        super(XOnlyNumpyArrayIterator, self).__init__(X.shape[0], batch_size, shuffle, seed)

    def next(self):
        # for python 2.x.
        # Keeps under lock only the mechanism which advances
        # the indexing of each batch
        # see http://anandology.com/blog/using-iterators-and-generators/
        with self.lock:
            index_array, current_index, current_batch_size = next(self.index_generator)
        # The transformation of images is not under thread lock so it can be done in parallel
        batch_x = np.zeros(tuple([current_batch_size] + list(self.X.shape)[1:]))
        for i, j in enumerate(index_array):
            x = self.X[j]
            x = self.image_data_generator.random_transform(x.astype('float32'))
            x = self.image_data_generator.standardize(x)
            batch_x[i] = x
        if self.save_to_dir:
            for i in range(current_batch_size):
                img = array_to_img(batch_x[i], self.dim_ordering, scale=True)
                fname = '{prefix}_{index}_{hash}.{format}'.format(prefix=self.save_prefix,
                                                                  index=current_index + i,
                                                                  hash=np.random.randint(1e4),
                                                                  format=self.save_format)
                img.save(os.path.join(self.save_to_dir, fname))
        return batch_x


class MultiYNumpyArrayIterator(Iterator):

    def __init__(self, X, y, image_data_generator,
                 batch_size=32, shuffle=False, seed=None,
                 dim_ordering='default',
                 save_to_dir=None, save_prefix='', save_format='jpeg'):
        if y is not None and len(X) != len(y):
            if isinstance(y, list):
                for y_sub in y:
                    if len(X) != len(y_sub):
                        raise ValueError('X (images tensor) and y (labels) '
                                         'should have the same length. '
                                         'Found: X.shape = %s, y.shape = %s' % (
                                         np.asarray(X).shape, np.asarray(y).shape))
            else:
                raise ValueError('X (images tensor) and y (labels) '
                                 'should have the same length. '
                                 'Found: X.shape = %s, y.shape = %s' % (np.asarray(X).shape, np.asarray(y).shape))
        if dim_ordering == 'default':
            dim_ordering = K.image_dim_ordering()
        self.X = np.asarray(X)
        if self.X.ndim != 4:
            raise ValueError('Input data in `NumpyArrayIterator` '
                             'should have rank 4. You passed an array '
                             'with shape', self.X.shape)
        channels_axis = 3 if dim_ordering == 'tf' else 1
        if self.X.shape[channels_axis] not in {1, 3, 4}:
            raise ValueError('NumpyArrayIterator is set to use the '
                             'dimension ordering convention "' + dim_ordering + '" '
                             '(channels on axis ' + str(channels_axis) + '), i.e. expected '
                             'either 1, 3 or 4 channels on axis ' + str(channels_axis) + '. '
                             'However, it was passed an array with shape ' + str(self.X.shape) +
                             ' (' + str(self.X.shape[channels_axis]) + ' channels).')
        if y is not None:
            if isinstance(y, list):
                self.y = [np.asarray(y_sub) for y_sub in y]
            else:
                self.y = np.asarray(y)
        else:
            self.y = None
        self.image_data_generator = image_data_generator
        self.dim_ordering = dim_ordering
        self.save_to_dir = save_to_dir
        self.save_prefix = save_prefix
        self.save_format = save_format
        super(MultiYNumpyArrayIterator, self).__init__(X.shape[0], batch_size, shuffle, seed)

    def next(self):
        # for python 2.x.
        # Keeps under lock only the mechanism which advances
        # the indexing of each batch
        # see http://anandology.com/blog/using-iterators-and-generators/
        with self.lock:
            index_array, current_index, current_batch_size = next(self.index_generator)
        # The transformation of images is not under thread lock so it can be done in parallel
        batch_x = np.zeros(tuple([current_batch_size] + list(self.X.shape)[1:]))
        for i, j in enumerate(index_array):
            x = self.X[j]
            x = self.image_data_generator.random_transform(x.astype('float32'))
            x = self.image_data_generator.standardize(x)
            batch_x[i] = x
        if self.save_to_dir:
            for i in range(current_batch_size):
                img = array_to_img(batch_x[i], self.dim_ordering, scale=True)
                fname = '{prefix}_{index}_{hash}.{format}'.format(prefix=self.save_prefix,
                                                                  index=current_index + i,
                                                                  hash=np.random.randint(1e4),
                                                                  format=self.save_format)
                img.save(os.path.join(self.save_to_dir, fname))
        if self.y is None:
            return batch_x
        if isinstance(self.y,list):
            batch_y = [y_sub[index_array] for y_sub in self.y]
        else:
            batch_y = self.y[index_array]

        return batch_x, batch_y

class MultiXYNumpyArrayIterator(Iterator):

    def __init__(self, X, y, image_data_generator,
                 batch_size=32, shuffle=False, seed=None,
                 dim_ordering='default',
                 save_to_dir=None, save_prefix='', save_format='jpeg',transforms=None):
        if y is not None and len(X) != len(y):
            if isinstance(y, list):
                for y_sub in y:
                    if len(X) != len(y_sub):
                        raise ValueError('X (images tensor) and y (labels) '
                                         'should have the same length. '
                                         'Found: X.shape = %s, y.shape = %s' % (
                                         np.asarray(X).shape, np.asarray(y).shape))
            else:
                raise ValueError('X (images tensor) and y (labels) '
                                 'should have the same length. '
                                 'Found: X.shape = %s, y.shape = %s' % (np.asarray(X).shape, np.asarray(y).shape))
        if dim_ordering == 'default':
            dim_ordering = K.image_dim_ordering()
        self.X = np.asarray(X)
        if self.X.ndim != 4:
            raise ValueError('Input data in `NumpyArrayIterator` '
                             'should have rank 4. You passed an array '
                             'with shape', self.X.shape)
        channels_axis = 3 if dim_ordering == 'tf' else 1
        if self.X.shape[channels_axis] not in {1, 3, 4}:
            raise ValueError('NumpyArrayIterator is set to use the '
                             'dimension ordering convention "' + dim_ordering + '" '
                             '(channels on axis ' + str(channels_axis) + '), i.e. expected '
                             'either 1, 3 or 4 channels on axis ' + str(channels_axis) + '. '
                             'However, it was passed an array with shape ' + str(self.X.shape) +
                             ' (' + str(self.X.shape[channels_axis]) + ' channels).')
        if y is not None:
            if isinstance(y, list):
                self.y = [np.asarray(y_sub) for y_sub in y]
            else:
                self.y = np.asarray(y)
        else:
            self.y = None
        self.image_data_generator = image_data_generator
        self.dim_ordering = dim_ordering
        self.save_to_dir = save_to_dir
        self.save_prefix = save_prefix
        self.save_format = save_format
        self.transforms = transforms
        super(MultiXYNumpyArrayIterator, self).__init__(X.shape[0], batch_size, shuffle, seed)

    def next(self):
        # for python 2.x.
        # Keeps under lock only the mechanism which advances
        # the indexing of each batch
        # see http://anandology.com/blog/using-iterators-and-generators/
        with self.lock:
            index_array, current_index, current_batch_size = next(self.index_generator)
        # The transformation of images is not under thread lock so it can be done in parallel
        batch_x = np.zeros(tuple([current_batch_size] + list(self.X.shape)[1:]))

        if isinstance(self.y, list):
            use_y = self.y[0]#batch_y = [y_sub[index_array] for y_sub in self.y]
        else:
            use_y = self.y#[index_array]

        batch_y = np.zeros(tuple([current_batch_size] + list(use_y.shape)[1:]))

        for i, j in enumerate(index_array):
            x = self.X[j]
            y = use_y[j]

            x,y = self.image_data_generator.random_transform(x.astype('float32'), y.astype('float32'))
            batch_x[i] = x
            batch_y[i] = y

        if self.save_to_dir:#todo handle y
            for i in range(current_batch_size):
                img = array_to_img(batch_x[i], self.dim_ordering, scale=True)
                fname = '{prefix}_{index}_{hash}.{format}'.format(prefix=self.save_prefix,
                                                                  index=current_index + i,
                                                                  hash=np.random.randint(1e4),
                                                                  format=self.save_format)
                img.save(os.path.join(self.save_to_dir, fname))
        #batch_y = batch_y.reshape((current_batch_size, -1))
        if isinstance(self.y, list):
            batch_y_out = [batch_y]
            for i in range(1, len(self.y)):
                batch_y_out.append(self.y[i][index_array])

            if self.transforms is not None:
                batch_y_out_new = []
                for i in range(0, len(self.transforms)):
                    batch_y_out_new.extend(self.transforms[i](batch_y_out))
                batch_y_out = batch_y_out_new

        else:
            batch_y_out = batch_y
            if self.transforms is not None:
                batch_y_out = self.transforms[0](batch_y_out)
        return batch_x,batch_y_out


class MultiYImageDataGenerator(ImageDataGenerator):

    def flow(self, X, y, batch_size=32, shuffle=True, seed=None,
             save_to_dir=None, save_prefix='', save_format='jpeg'):
        return MultiYNumpyArrayIterator(
            X, y, self,
            batch_size=batch_size, shuffle=shuffle, seed=seed,
            dim_ordering=self.dim_ordering,
            save_to_dir=save_to_dir, save_prefix=save_prefix, save_format=save_format)

class MultiXYImageDataGenerator(ImageDataGenerator):

    def __init__(self, transforms=None, **super_kwargs):
        self.transforms  = transforms
        super(MultiXYImageDataGenerator, self).__init__(**super_kwargs)


    def flow(self, X, y, batch_size=32, shuffle=True, seed=None,
             save_to_dir=None, save_prefix='', save_format='jpeg'):
        return MultiXYNumpyArrayIterator(
            X, y, self,
            batch_size=batch_size, shuffle=shuffle, seed=seed,
            dim_ordering=self.dim_ordering,
            save_to_dir=save_to_dir, save_prefix=save_prefix, save_format=save_format, transforms=self.transforms)

    def random_transform(self, x, y):
        # x is a single image, so it doesn't have image number at index 0
        img_row_index = self.row_index - 1
        img_col_index = self.col_index - 1
        img_channel_index = self.channel_index - 1

        # use composition of homographies to generate final transform that needs to be applied
        if self.rotation_range:
            theta = np.pi / 180 * np.random.uniform(-self.rotation_range, self.rotation_range)
        else:
            theta = 0
        rotation_matrix = np.array([[np.cos(theta), -np.sin(theta), 0],
                                    [np.sin(theta), np.cos(theta), 0],
                                    [0, 0, 1]])
        if self.height_shift_range:
            tx = np.random.uniform(-self.height_shift_range, self.height_shift_range) * x.shape[img_row_index]
        else:
            tx = 0

        if self.width_shift_range:
            ty = np.random.uniform(-self.width_shift_range, self.width_shift_range) * x.shape[img_col_index]
        else:
            ty = 0

        translation_matrix = np.array([[1, 0, tx],
                                       [0, 1, ty],
                                       [0, 0, 1]])
        if self.shear_range:
            shear = np.random.uniform(-self.shear_range, self.shear_range)
        else:
            shear = 0
        shear_matrix = np.array([[1, -np.sin(shear), 0],
                                 [0, np.cos(shear), 0],
                                 [0, 0, 1]])

        if self.zoom_range[0] == 1 and self.zoom_range[1] == 1:
            zx, zy = 1, 1
        else:
            zx, zy = np.random.uniform(self.zoom_range[0], self.zoom_range[1], 2)
        zoom_matrix = np.array([[zx, 0, 0],
                                [0, zy, 0],
                                [0, 0, 1]])

        transform_matrix = np.dot(np.dot(np.dot(rotation_matrix, translation_matrix), shear_matrix), zoom_matrix)

        h, w = x.shape[img_row_index], x.shape[img_col_index]
        transform_matrix = transform_matrix_offset_center(transform_matrix, h, w)
        print(x.shape)
        x = apply_transform(x, transform_matrix, img_channel_index,
                            fill_mode=self.fill_mode, cval=self.cval)

        print(y.shape)
        y =apply_transform(y, transform_matrix, img_channel_index,
                            fill_mode=self.fill_mode, cval=self.cval)

        if self.channel_shift_range != 0:
            x = random_channel_shift(x, self.channel_shift_range, img_channel_index)

        # don't do y channel shift

        if self.horizontal_flip:
            if np.random.random() < 0.5:
                x = flip_axis(x, img_col_index)
                y = flip_axis(y, img_col_index)

        if self.vertical_flip:
            if np.random.random() < 0.5:
                x = flip_axis(x, img_row_index)
                y = flip_axis(y, img_row_index)

        # TODO:
        # channel-wise normalization
        # barrel/fisheye
        return x, y



class OmniImageDataGenerator(ImageDataGenerator):

    def __init__(self, transforms=None, **super_kwargs):
        self.transforms  = transforms
        super(OmniImageDataGenerator, self).__init__(**super_kwargs)


    def flow(self, X, y, batch_size=32, shuffle=True, seed=None,
             save_to_dir=None, save_prefix='', save_format='jpeg'):
        """X and y are expected to be  tuple lists of the form [(data,transform)] where transform
        indicates whether or not to transform it """

        return OmniNumpyArrayIterator(
            X, y, self,
            batch_size=batch_size, shuffle=shuffle, seed=seed,
            dim_ordering=self.dim_ordering,
            save_to_dir=save_to_dir, save_prefix=save_prefix, save_format=save_format, transforms=self.transforms)

    def random_transform(self, x, y):
        x_shape = None
        for x_sub in x:
            if x_sub[1]:
                x_shape = x_sub.shape

        img_channel_index, img_col_index, img_row_index, transform_matrix = self.build_transform(x_shape)
        flip_horiz_choice, flip_vert_choice = np.random.random(2) < 0.5
        for x_i in range(len(x)):
            if x[x_i][1]:
                to_transform = x[x_i][0]
                to_transform = self.spatial_transforms(flip_horiz_choice, flip_vert_choice, img_channel_index,
                                                       img_col_index, img_row_index, to_transform, transform_matrix)

                if self.channel_shift_range != 0:
                    to_transform = random_channel_shift(to_transform, self.channel_shift_range, img_channel_index)

                x[x_i][0] = to_transform
        if y is not None:
            for y_i in range(len(y)):
                if y[y_i][1]:
                    to_transform = y[y_i][0]
                    to_transform = self.spatial_transforms(flip_horiz_choice, flip_vert_choice, img_channel_index,
                                                           img_col_index, img_row_index, to_transform, transform_matrix)

                    y[y_i][0] = to_transform

        # TODO:
        # channel-wise normalization
        # barrel/fisheye
        return x, y

    def spatial_transforms(self, flip_horiz_choice, flip_vert_choice, img_channel_index, img_col_index, img_row_index,
                           to_transform, transform_matrix):
        to_transform = apply_transform(to_transform, transform_matrix, img_channel_index,
                                       fill_mode=self.fill_mode, cval=self.cval)
        if self.horizontal_flip:
            if flip_horiz_choice:
                to_transform = flip_axis(to_transform, img_col_index)
        if self.vertical_flip:
            if flip_vert_choice < 0.5:
                to_transform = flip_axis(to_transform, img_row_index)
        return to_transform

    def build_transform(self, shape):
        img_row_index = self.row_index - 1
        img_col_index = self.col_index - 1
        img_channel_index = self.channel_index - 1
        # use composition of homographies to generate final transform that needs to be applied
        if self.rotation_range:
            theta = np.pi / 180 * np.random.uniform(-self.rotation_range, self.rotation_range)
        else:
            theta = 0
        rotation_matrix = np.array([[np.cos(theta), -np.sin(theta), 0],
                                    [np.sin(theta), np.cos(theta), 0],
                                    [0, 0, 1]])
        if self.height_shift_range:
            tx = np.random.uniform(-self.height_shift_range, self.height_shift_range) * shape[img_row_index]
        else:
            tx = 0
        if self.width_shift_range:
            ty = np.random.uniform(-self.width_shift_range, self.width_shift_range) * shape[img_col_index]
        else:
            ty = 0
        translation_matrix = np.array([[1, 0, tx],
                                       [0, 1, ty],
                                       [0, 0, 1]])
        if self.shear_range:
            shear = np.random.uniform(-self.shear_range, self.shear_range)
        else:
            shear = 0
        shear_matrix = np.array([[1, -np.sin(shear), 0],
                                 [0, np.cos(shear), 0],
                                 [0, 0, 1]])
        if self.zoom_range[0] == 1 and self.zoom_range[1] == 1:
            zx, zy = 1, 1
        else:
            zx, zy = np.random.uniform(self.zoom_range[0], self.zoom_range[1], 2)
        zoom_matrix = np.array([[zx, 0, 0],
                                [0, zy, 0],
                                [0, 0, 1]])
        transform_matrix = np.dot(np.dot(np.dot(rotation_matrix, translation_matrix), shear_matrix), zoom_matrix)
        h, w = shape[img_row_index], shape[img_col_index]
        transform_matrix = transform_matrix_offset_center(transform_matrix, h, w)
        return img_channel_index, img_col_index, img_row_index, transform_matrix


class OmniNumpyArrayIterator(Iterator):

    def __init__(self, X, y, image_data_generator,
                 batch_size=32, shuffle=False, seed=None,
                 dim_ordering='default',
                 save_to_dir=None, save_prefix='', save_format='jpeg',transforms=None):

        if dim_ordering == 'default':
            dim_ordering = K.image_dim_ordering()

        self.X = X#np.asarray(X)
        for x_i in range(len(self.X)):
            sub_x = np.asarray(self.X[x_i][0], dtype=np.float32)
            self.X[x_i][0] = sub_x
            if sub_x != 4:
                raise ValueError('Input data in `OmniNumpyArrayIterator` '
                                 'should have rank 4. You passed an array {0}'
                                 'with shape {1}'.format(x_i,sub_x.shape))
            channels_axis = 3 if dim_ordering == 'tf' else 1
            if sub_x.shape[channels_axis] not in {1, 3, 4}:
                raise ValueError('OmniNumpyArrayIterator is set to use the '
                                 'dimension ordering convention "' + dim_ordering + '" '
                                 '(channels on axis ' + str(channels_axis) + '), i.e. expected '
                                 'either 1, 3 or 4 channels on axis ' + str(channels_axis) + '. '
                                 'However, it was passed an array at ' + str(x_i) + ' with shape ' + str(sub_x.shape) +
                                 ' (' + str(sub_x.shape[channels_axis]) + ' channels).')
        self.y = y
        if y is not None:
            for y_i in range(len(self.y)):
                sub_x = np.asarray(self.y[y_i][0], dtype=np.float32)
                self.y[y_i][0] = sub_y

        self.image_data_generator = image_data_generator
        self.dim_ordering = dim_ordering
        self.save_to_dir = save_to_dir
        self.save_prefix = save_prefix
        self.save_format = save_format
        self.transforms = transforms
        super(OmniNumpyArrayIterator, self).__init__(X[0][0].shape[0], batch_size, shuffle, seed)

    def next(self):
        # for python 2.x.
        # Keeps under lock only the mechanism which advances
        # the indexing of each batch
        # see http://anandology.com/blog/using-iterators-and-generators/
        with self.lock:
            index_array, current_index, current_batch_size = next(self.index_generator)
        # The transformation of images is not under thread lock so it can be done in parallel

        result_x = [[] for x in self.X]
        if self.y is not None:
            result_y = [[] for y in self.y]

        for j in enumerate(index_array):
            tmp_x = [(x_sub[0][j], x_sub[1]) for x_sub in self.X]
            if self.y is not None:
                tmp_y = [(y_sub[0][j], y_sub[1]) for y_sub in self.y]
            else:
                tmp_y = None
            x,y = self.image_data_generator.random_transform(tmp_x, tmp_y)
            for x_i in range(len(x)):
                result_x[x_i].append(x[x_i][0].reshape((1,)+x[x_i][0].shape))
            if self.y is not None:
                for y_i in range(len(y)):
                    result_y[y_i].append(y[y_i][0].reshape((1,)+y[y_i][0].shape))

        for x_i in range(len(x)):
            result_x[x_i]  = np.concatenate(result_x[x_i],axis=0)

        if self.y is not None:
            for y_i in range(len(y)):
                result_y[y_i] = np.concatenate(result_y[y_i],axis=0)

        if self.save_to_dir:#todo handle y
            for i in range(current_batch_size):
                img = array_to_img(result_x[i], self.dim_ordering, scale=True)
                fname = '{prefix}_{index}_{hash}.{format}'.format(prefix=self.save_prefix,
                                                                  index=current_index + i,
                                                                  hash=np.random.randint(1e4),
                                                                  format=self.save_format)
                img.save(os.path.join(self.save_to_dir, fname))

        if self.y is not None:

            if self.transforms is not None:
                for y_i in range(len(y)):
                    result_y[y_i] = self.transforms[y_i](result_y[y_i])
        return result_x,result_y
