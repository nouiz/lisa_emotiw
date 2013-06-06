import numpy
import warnings

class FaceTable(object):
    """
    Faces table that indicates the location of the faces and nonfaces.
    """
    def __init__(self, ntotal_examples, n_total_faces, n_total_nonfaces, face_nonface_ratio=0.5):
        self.ntotal_examples = ntotal_examples
        self.n_total_faces = n_total_faces
        self.n_total_nonfaces = n_total_nonfaces
        self.face_nonface_ratio = face_nonface_ratio
        #First column: Flag for face.
        #Second column: Flag for non-face.
        #Third column: The row that image is in its own dataset
        #Fourth column: Did it survive or not?
        self.face_table = numpy.zeros((ntotal_examples, 4))
        #This holds a pointer to the empty patch index:
        self.last_empty_idx = 0

    def compute_face_nonface_ratio(self):
        faces = self.get_survived_faces()
        nonfaces = self.get_survived_nonfaces()

        n_faces = faces.shape[0]
        n_nonfaces = nonfaces.shape[0]

        ratio = n_faces / float(n_nonfaces)

        return ratio

    def get_survived_faces(self):
        survived_faces = numpy.where((self.face_table[:, 0] != 0) & (self.face_table[:, 3] != 0))
        return survived_faces[0]

    def get_survived_nonfaces(self):
        survived_faces = numpy.where((self.face_table[:, 1] != 0) & (self.face_table[:, 3] != 0))
        return survived_faces[0]

    def get_dead_faces(self):
        survived_faces = numpy.where((self.face_table[:, 0] != 0) & (self.face_table[:, 3] != 1))
        return survived_faces[0]

    def get_dead_nonfaces(self):
        survived_nonfaces = numpy.where((self.face_table[:, 1] != 0) & (self.face_table[:, 3] != 1))
        return survived_nonfaces[0]

    def get_survived_faces_imgnos(self):
        survived_faces = self.get_survived_faces()
        return self.face_table[survived_faces[0], 3]

    def get_survived_nonfaces_imgnos(self):
        survived_nonfaces = self.get_survived_nonfaces()
        return self.face_table[survived_nonfaces[0], 3]

    def get_dead_faces_imgnos(self):
        dead_faces = self.get_dead_faces()
        return self.face_table[dead_faces[0], 3]

    def get_dead_nonfaces_imgnos(self):
        dead_nonfaces = self.get_dead_nonfaces()
        return self.face_table[dead_nonfaces[0], 3]

    def update_face_table(self, faceno, nonfaceno, is_face):
        """
        If table has a face insert the non_face into it. Otherwise put the nonface in it.
        """
        col_no = 0 if is_face else 1
        other_col_no = 1 - col_no

        img_no = faceno if is_face else nonfaceno
        new_img_no = faceno if not is_face else nonfaceno

        face_rows = numpy.where((self.face_table[:, col_no] != 0) & (self.face_table[:, 2] == img_no))
        assert face_rows[0].ndim == 1
        face_row = face_rows[0][0]
        self.face_table[face_row, col_no] = 0
        self.face_table[face_row, other_col_no] = 1
        self.face_table[face_row, 2] = new_img_no
        self.face_table[face_row, 3] = 1

    def kill_face_table(self, imgno, is_face):
        """
        Makes the survival state of the given image 1
        """
        col_no = 0 if is_face else 1
        face_rows = numpy.where((self.face_table[:, col_no] != 0) & (self.face_table[:, 2] == imgno))
        assert face_row[0].ndim == 1
        face_row = face_row[0][0]
        self.face_table[face_row, 3] = 0

    def insert_face_table(self, imgno, is_face):
        """
        This one adds a new face to the table.
        """
        col_no = 0 if is_face else 1
        self.face_table[self.last_empty_idx, col_no] = 1
        self.face_table[self.last_empty_idx, 2] = imgno
        self.face_table[self.last_empty_idx, 3] = 1

    def check_table_balance_ratio(self, face_ratio_down_limit=0.45, face_ratio_up_limit=0.6)):
        face_ratio = self.compute_face_nonface_ratio()
        if face_ratio <= face_ratio_limit:
            warnings.warn("Face ratio is %.2f below the limits." % face_ratio_limit)
            return False
        else:
            if face_ratio >= face_ratio_up_limit:
                warnings.warn("Face ratio is too large.")
            else:
                print "Face ratio is in acceptable level."
            return True

    def balance_table(self):
        """
        This algorithm finds the dead instances and replaces them with the alive ones.
        """
        self.check_table_balance_ratio()

