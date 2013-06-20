import numpy
import warnings


class FaceTable(object):
    """
    Faces table that indicates the location of the faces and nonfaces.
    """
    def __init__(self,
            ntotal_examples,
            n_total_faces,
            n_total_nonfaces,
            is_filled=False,
            face_nonface_ratio=0.5):

        self.is_filled = is_filled
        self.ntotal_examples = ntotal_examples
        self.n_total_faces = n_total_faces
        self.n_total_nonfaces = n_total_nonfaces
        self.face_nonface_ratio = face_nonface_ratio

        #First column: Flag for face. 1 if it is face 0 if it is not.
        #Second column: The row that image is in its own dataset
        #Third column: Could the face survive or not?
        #0 if it is dead. 1 if it is alive.
        self.face_table = numpy.zeros((ntotal_examples, 3))

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
        survived_faces = numpy.where((self.face_table[:, 0] == 1) & (self.face_table[:, 2] == 1))
        return survived_faces[0]

    def get_survived_nonfaces(self):
        survived_faces = numpy.where((self.face_table[:, 0] == 0) & (self.face_table[:, 2] == 1))
        return survived_faces[0]

    def get_dead_faces(self):
        survived_faces = numpy.where((self.face_table[:, 0] == 1) & (self.face_table[:, 2] == 0))
        return survived_faces[0]

    def get_dead_nonfaces(self):
        survived_nonfaces = numpy.where((self.face_table[:, 0] == 0) & (self.face_table[:, 2] == 0))
        return survived_nonfaces[0]

    def get_survived_faces_imgnos(self):
        survived_faces = self.get_survived_faces()
        return self.face_table[survived_faces[0], 1]

    def get_survived_nonfaces_imgnos(self):
        """
        Return the nonface images that were able to survive.
        """
        survived_nonfaces = self.get_survived_nonfaces()
        return self.face_table[survived_nonfaces[0], 1]

    def get_dead_faces_imgnos(self):
        """
        Return the dead face images.
        """
        dead_faces = self.get_dead_faces()
        return self.face_table[dead_faces[0], 1]

    def get_dead_nonfaces_imgnos(self):
        """
        Return the dead nonface images.
        """
        dead_nonfaces = self.get_dead_nonfaces()
        return self.face_table[dead_nonfaces[0], 1]

    def get_last_face_no(self):
        """
        Get the last face in the table.
        """
        faces = numpy.where((self.face_table[:, 0]==1))[0]
        return numpy.max(self.face_table[faces, 1])

    def get_last_nonface_no(self):
        """
        If table has a face insert the non_face into it. Otherwise put the nonface in it.
        """
        dead_nonfaces = numpy.where((self.face_table[:, 1]==0))[0]
        return numpy.max(self.face_table[dead_nonfaces, 1])

    def update_face_table(self,
            old_faceno,
            new_faceno,
            old_face_status,
            new_face_status):
        """
            If table has a face update it to the non_face into it. Otherwise put the nonface in it.
        """
        face_rows = numpy.where((self.face_table[: , 0] == old_face_status) & (self.face_table[: , 1] == old_faceno))

        assert face_rows[0].ndim == 1

        face_row = face_rows[0][0]
        self.face_table[face_row, 0] = new_face_status
        self.face_table[face_row, 1] = new_faceno
        self.face_table[face_row, 2] = 1

    def simple_update_face_table(self,
            img_row,
            img_no,
            is_face,
            is_alive=1):
        """
            If table has a face update it to the non_face into it. Otherwise put the nonface in it.
        """
        self.face_table[img_row, 0] = is_face
        self.face_table[img_row, 1] = img_no
        self.face_table[img_row, 2] = is_alive

    def get_row(self, row_no):
        face_props = self.face_table[row_no]
        is_face = False
        if face_props[0] == 0:
            is_face = True
        imgno = face_props[1]
        is_alive = face_props[2]
        return (imgno, is_face, is_alive)

    def kill_face_table(self, imgno, is_face):
        """
        Makes the survival state of the given image 1
        """
        face_rows = numpy.where((self.face_table[:, 0] == is_face) & (self.face_table[:, 1] == imgno))
        assert face_rows[0].ndim == 1
        face_row = face_rows[0][0]
        self.face_table[face_row, 2] = 0

    def insert_face_table(self, imgno, is_face):
        """
        This one adds a new face to the table.
        """
        assert self.last_empty_idx < self.ntotal_examples
        self.face_table[self.last_empty_idx, 0] = is_face
        self.face_table[self.last_empty_idx, 1] = imgno
        self.face_table[self.last_empty_idx, 2] = 1
        self.last_empty_idx += 1

    def check_table_balance_ratio(self, face_ratio_down_limit=0.45, face_ratio_up_limit=0.6):
        face_ratio = self.compute_face_nonface_ratio()
        if face_ratio <= face_ratio_down_limit:
            warnings.warn("Face ratio is %.2f below the limits." % face_ratio_down_limit)
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

