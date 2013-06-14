# Copyright (c) 2012--2013 University of Montreal, Pascal Vincent, Pascal Lamblin
#
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
#     * Redistributions of source code must retain the above copyright
#       notice, this list of conditions and the following disclaimer.
#     * Redistributions in binary form must reproduce the above copyright
#       notice, this list of conditions and the following disclaimer in the
#       documentation and/or other materials provided with the distribution.
#     * The names of the authors and contributors to this software may not be
#       used to endorse or promote products derived from this software without
#       specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ''AS IS'' AND ANY
# EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
# WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDERS BE LIABLE FOR ANY
# DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
# (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
# LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
# ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
# (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
# SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.


import os

def locate_data_path(filename, path_starts=None):
    """
        looks for filename under the various options listed in path_starts
    """
    # TO DO: look up DATAPTH environment variable to include its content to the path_starts

    if os.path.exists(filename):
            return filename

    if path_starts is None:
        path_starts = [
                os.path.join(os.path.expanduser("~"), "data"),
                "/data/lisa/data"]
    for start in path_starts:
        path = os.path.join(start, filename)
        if os.path.exists(path):
            return path
    raise IOError("Could not locate file or directory "+filename+" in "+str(path_starts))


def search_replace(str, search_replace_dict):
    """Performs, on str, all search-replace substitutions listed as key->value pairs in the given dictionary""" 
    for search_str, replace_str in search_replace_dict.items():
        str = str.replace(search_str, replace_str)
    return str
