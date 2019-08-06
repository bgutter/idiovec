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

from . import cruft

class Identifier( object ):
    """
    Implements an identification algorithm for some author. Identifiers can be constructed
    by adding texts of some user together, and then compared to one another.
    """

    def sampleTexts( texts ):
        """
        Update instance of this identifier with additional sample texts for the author.
        """
        raise NotImplementedError

    def compare( other ):
        """
        Return a distance between 0.0 and 1.0 between this Identifier and another of the same
        subclass.
        """
        raise NotImplementedError

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

    def sampleTexts( self, texts ):
        """
        Update length distribution with new texts.
        """
        for t in texts:
            # TODO normalize after loop
            h, _ = np.histogram( [ len( w ) for w in nltk.word_tokenize( t ) ], bins=self.bin_edges )
            self.dist = ( ( self.sample_count * self.dist ) + h ) / ( self.sample_count + 1 )
            self.sample_count += 1

    def compare( self, other ):
        """
        Get the distance between this MendenhallIdentifier and another.
        """
        assert( all( self.bin_edges == other.bin_edges ) )
        ks, p = sp.stats.ks_2samp( self.dist, other.dist )
        return ks

    def plot( self ):
        plt.figure()
        plt.plot( self.dist )
        plt.show()

def Idiovec( object ):
    """
    Implements idiovecs for a given author or source.
    """

    def __init__( self ):
        """
        Initialize a new Idiovec.
        """
        self.identifiers = []
        self.identifiers.append( MendenhallIdentifier() )

# TEST DRIVER

import pickle

authors = [ "GallowBoob", "Unidan" ]
texts = {}

for a in authors:
    json = requests.get( "https://api.pushshift.io/reddit/search/comment/?author={}".format( a ) ).json()
    texts[ a ] = [ x[ 'body' ] for x in json[ "data" ] ]

m1 = MendenhallIdentifier()
m1.sampleTexts( texts[ "GallowBoob" ] )

m2 = MendenhallIdentifier()
m2.sampleTexts( texts[ "Unidan" ] )

d = m1.compare( m2 )

m2.plot()
