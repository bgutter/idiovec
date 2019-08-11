"""
idiovec.hy

Construct idiolect vectors from a collection of messages from a variety of users.

Common Terminology
|------------+--------------------------------------------------------------------------------------------------------------|
| Text       | A single body of text from some source (tweet, comment, text, email, etc)                                    |
| Source     | Technical origin of a text (twitter account, reddit account, email address, etc)                             |
| Author     | Human origin of a text.                                                                                      |
| Identifier | Some pattern in an author's texts that discerns them from other authors.                                     |
| Language   | A method of encoding ideas that's mutually intelligible to some community.                                   |
| Dialect    | A subset of a language that has different patterns than the overall population of speakers of that language. |
| Idiolect   | The deviations in the use of a language (or dialect) applicable to a single author.                          |
| Idiovec    | A vectorized expression of an idiolect, useful for detecting source duplication by a single author at scale. |
|------------+--------------------------------------------------------------------------------------------------------------|
"""

import nltk
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
import requests

import tensorflow as tf

class Identifier( object ):
    """
    Implements an identification algorithm for some author. Identifiers can be constructed
    by adding texts of some user together, and then compared to one another.
    """

    def sample_texts( self, texts ):
        """
        Update instance of this identifier with additional sample texts for the author.
        """
        raise NotImplementedError

    @property
    def value( self ):
        """
        Return a (1,M) shaped vector of floats describing this identifier.
        """
        raise NotImplementedError

    #def compare( other ):
    #    """
    #    Return a distance between 0.0 and 1.0 between this Identifier and another of the same
    #    subclass.
    #    """
    #    raise NotImplementedError

class MendenhallIdentifier( Identifier ):
    """
    This implements Mendenhall's Characteristic Curve of Composition.

    Mendenhall hypothesized that authors could be attributed based on the distribution of lengths
    of words in their works. Literally, words are transformed into their character length, and then
    the resulting frequency distribution over those lengths is used as an identifier.

    Mendenhall defined no method for comparing the distributions. Here, we use a KS test.
    """

    def __init__( self, bins=(0,20,20) ):
        """
        Initialize a new MendenhallIdentifier.
            bins: Arguments to np.linspace(...) which generates bin edges.
        """
        self.bin_edges    = np.linspace( *bins )
        self.dist         = np.zeros( len( self.bin_edges ) - 1 )
        self.sample_count = 0

    def sample_texts( self, texts ):
        """
        Update length distribution with new texts.
        """
        for t in texts:
            # TODO normalize after loop
            words = nltk.word_tokenize( t )
            if len( words ) > 0:
                h, _ = np.histogram( [ len( w ) for w in words ], bins=self.bin_edges )
                self.dist = ( ( self.sample_count * self.dist ) + h ) / ( self.sample_count + 1 )
                self.sample_count += 1

    @property
    def value( self ):
        """
        Current dist is our value.
        """
        return self.dist

    #def compare( self, other ):
    #    """
    #    Get the distance between this MendenhallIdentifier and another.
    #    """
    #    assert( all( self.bin_edges == other.bin_edges ) )
    #    ks, p = sp.stats.ks_2samp( self.dist, other.dist )
    #    return ks

    def plot( self ):
        plt.figure()
        plt.plot( self.dist )
        plt.show()

class IdiovecModel( object ):
    """
    Fits idiolect vectors to a training set.
    """

    def __init__( self, dim=8 ):
        """
        Initialize a new Idiovec.
            dim: Dimensionality of the idiolect embeddings.
        """
        # Model Parameters
        self.dim = dim
        self.identifierClasses = []
        self.identifierClasses.append( MendenhallIdentifier )

        # State
        self.stylometric_characteristic_vectors = []
        self.label_indexes = []
        self.label_record = []

    def sample_texts( self, texts, labels ):
        """
        Add more sample texts.
            texts:  A vector of texts of length M.
            labels: A vector of labels of length M -- for example, authors.
        """
        assert( len( texts ) == len( labels ) )
        for text, label in zip( texts, labels ):
            stylometric_vector = []
            for ic in self.identifierClasses:
                idt = ic()
                #
                # TODO: This sucks. Mendenhall's value is basically useless with a single message.
                #       Could potentially use boosting? Or send texts in batches to the identifiers?
                #
                idt.sample_texts( [ text ] )
                stylometric_vector.extend( idt.value )
            self.stylometric_characteristic_vectors.append( np.array( stylometric_vector ) )
            if label not in self.label_record:
                self.label_record.append( label )
            self.label_indexes.append( self.label_record.index( label ) )

    def fit( self ):
        """
        Fit model to all sampled texts.
        """
        #
        # % Idiolect Embeddings %
        #
        # Input:  One-hot encoding of training label set.
        # Output: Catenation of stylometric vectors
        #
        # The hidden layer will contain embedding vectors for each input label. That's what we really want.
        #
        # Construct training data
        def onehot( idx ):
            assert( idx < len( self.label_record ) )
            ret = np.zeros( len( self.label_record ) )
            ret[ idx ] = 1.0
            return ret
        X_train = np.array( [ onehot( li ) for li in self.label_indexes ] )
        Y_train = np.array( self.stylometric_characteristic_vectors )
        assert( ( ~np.isnan( X_train ) ).all() )
        assert( ( ~np.isnan( Y_train ) ).all() )

        # Construct TensorFlow model
        DIM = 10
        training_output_layer = tf.placeholder( tf.float32, shape=( None, Y_train.shape[ 1 ] ) ) # predicted stylometric elements
        input_layer           = tf.placeholder( tf.float32, shape=( None, X_train.shape[ 1 ] ) ) # one-hot label mapping
        input_weights         = tf.Variable( tf.random_normal( [ X_train.shape[ 1 ], DIM ] ) )
        input_bias            = tf.Variable( tf.random_normal( [ DIM ] ) )
        embedding_layer       = tf.add( tf.matmul( input_layer, input_weights ), input_bias )
        prediction_weights    = tf.Variable( tf.random_normal( [ DIM, Y_train.shape[1] ] ) )
        prediction_bias       = tf.Variable( tf.random_normal( [ Y_train.shape[1] ] ) )
        prediction_layer      = tf.nn.softmax( tf.add( tf.matmul( embedding_layer, prediction_weights), prediction_bias ) )

        # Loss function is cosine distance...for now...
        loss_function = tf.losses.cosine_distance( tf.nn.l2_normalize( prediction_layer, 0 ),
                                                   tf.nn.l2_normalize( training_output_layer, 0 ),
                                                   dim=0 )

        # Use Adam for now
        train_step = tf.train.AdamOptimizer( 0.1 ).minimize( loss_function )

        # Fire up TensorFlow
        sess = tf.Session()
        init = tf.global_variables_initializer()
        sess.run( init )

        # Train it!
        for _ in range( 100 ):
            feed_dict = { input_layer: X_train, training_output_layer: Y_train }
            sess.run( train_step, feed_dict=feed_dict )
            dist = sess.run( loss_function, feed_dict=feed_dict )
            print( 'Loss is : ', dist )

        # Save the embeddings
        self._saveEmbeddings( sess.run( input_weights + input_bias ) )

        #
        # % Styolmetric Encoder %
        #
        # Input:  Catenation of stylometric vectors
        # Output: Embedding vector for that input.
        #
        

        #
        # Create a TensorFlow model with each X[n].value as an input, and the output vector embeddings.
        # Code adapted from: https://www.tensorflow.org/beta/tutorials/text/word_embeddings
        #  https://adventuresinmachinelearning.com/word2vec-keras-tutorial/
        #uniqueLabelCount = len( set( self.label_indexes ) )
        #catenatedIdentifierLength = self.stylometric_characteristic_vectors[0].dim[0]
        #model = keras.Sequential( [
        #    layers.Embedding( uniqueLabelCount, self.dim, catenatedIdentifierLength ),
        #    layers.Dense( 16, activation='relu' ),
        #    layers.Dense( 1, activation='sigmoid' )
        #    ])

        #model.compile(optimizer='adam',
        #      loss='cosine_distance',
        #              metrics=????)

    def _saveEmbeddings( self, embeddings ):
        """
        Save to CWD + embeddings.pickle for debugging.
        """
        import pickle
        var = { label: embedding for label, embedding in zip( self.label_record, embeddings ) }
        with open( "./embeddings.pickle", "wb" ) as fout:
            pickle.dump( var, fout )

    def transform( self, text ):
        """
        Given a text, calculate identifiers and return an idiovec. Requires
        fit() to have already been called.
        """
        raise NotImplementedError

# TEST DRIVER

if __name__ == "__main__":
    import pickle

    authors = [ "GallowBoob", "Unidan", "_vargas_" ]
    texts = {}

    for a in authors:
        json = requests.get( "https://api.pushshift.io/reddit/search/comment/?author={}".format( a ) ).json()
        texts[ a ] = [ x[ 'body' ] for x in json[ "data" ] ]

    mendenhalls = []
    for author in authors:
        l = len( texts[ author ] )
        left, right = texts[ author ][ : l // 2 ], texts[ author ][ l // 2 :]
        lm = MendenhallIdentifier()
        lm.sampleTexts( left )
        mendenhalls.append( lm )
        rm = MendenhallIdentifier()
        rm.sampleTexts( right )
        mendenhalls.append( rm )


    #m2.plot()
