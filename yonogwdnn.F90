! ==================================================================================================
!> @brief Module storing all persistent data for neural network functionality.
!
! Note that all bias vectors have a KLON dimension, along with all values are identical. This is to
! facilitate batched evaluations of the neural network.
!
!> @author
!> Matthew Chantry, University of Oxford (matthew.chantry@physics.ox.ac.uk)
!> Sam Hatfield, ECMWF (samuel.hatfield@ecmwf.int)
! ==================================================================================================

MODULE YONOGWDNN
    USE PARKIND1, ONLY: NNP

    IMPLICIT NONE
    
    ! Weights of input layer (NWIDTH, NINP)
    REAL(KIND=NNP), ALLOCATABLE :: INPUT(:,:)
  
    ! Weights of hidden layers (NWIDTH, NWIDTH, NHIDDEN)
    REAL(KIND=NNP), ALLOCATABLE :: HIDDEN(:,:,:)
  
    ! Weights of output layer (NOUT, NWIDTH)
    REAL(KIND=NNP), ALLOCATABLE :: OUTPUT(:,:)
  
    ! Biases of input layer (NWIDTH, KLON)
    REAL(KIND=NNP), ALLOCATABLE :: INPUT_B(:,:)
  
    ! Biases of hidden layers (NWIDTH, HIDDEN, KLON)
    REAL(KIND=NNP), ALLOCATABLE :: HIDDEN_B(:,:,:)
  
    ! Biases of output layer (NOUT, KLON)
    REAL(KIND=NNP), ALLOCATABLE :: OUTPUT_B(:,:)
  
    ! Width of the input layer
    INTEGER, PARAMETER :: NINP    = 5

    ! Width of each hidden layer
    INTEGER, PARAMETER :: NWIDTH  = 3

    ! Number of hidden layers
    INTEGER, PARAMETER :: NHIDDEN = 6

    ! Width of the output layer
    INTEGER, PARAMETER :: NOUT    = 5
END MODULE YONOGWDNN

