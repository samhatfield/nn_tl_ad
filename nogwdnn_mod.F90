! ==================================================================================================
!> @brief Module containing all neural network functionality.
!
! This module contains all subroutines for evaluating the neural network and its tangent-linear and
! adjoint. It is very similar to the actual subroutine used in the IFS.
!
!> @author
!> Matthew Chantry, University of Oxford (matthew.chantry@physics.ox.ac.uk)
!> Sam Hatfield, ECMWF (samuel.hatfield@ecmwf.int)
! ==================================================================================================

MODULE NOGWDNN_MOD

USE PARKIND1, ONLY : SING, DOUB, NNP
USE YONOGWDNN

IMPLICIT NONE

!> Generic precision-agnostic interface for BLAS matrix-matrix multiply.
!> See BLAS documentation for description of the arguments.
PUBLIC :: GEMM
INTERFACE GEMM
    SUBROUTINE DGEMM(TRANSA, TRANSB, M, N, K, ALPHA, A, LDA, B, LDB, BETA, C, LDC)
        USE PARKIND1, ONLY: DOUB
        CHARACTER, INTENT(IN) :: TRANSA
        CHARACTER, INTENT(IN) :: TRANSB
        INTEGER, INTENT(IN) :: M, N, K
        REAL(KIND=DOUB), INTENT(IN) :: ALPHA
        REAL(KIND=DOUB), INTENT(IN), DIMENSION(LDA,*) :: A
        INTEGER, INTENT(IN) :: LDA
        REAL(KIND=DOUB), INTENT(IN), DIMENSION(LDB,*) :: B
        INTEGER, INTENT(IN) :: LDB
        REAL(KIND=DOUB), INTENT(IN) :: BETA
        REAL(KIND=DOUB), INTENT(IN), DIMENSION(LDC,*) :: C
        INTEGER, INTENT(IN) :: LDC
    END SUBROUTINE DGEMM
    SUBROUTINE SGEMM(TRANSA, TRANSB, M, N, K, ALPHA, A, LDA, B, LDB, BETA, C, LDC)
        USE PARKIND1, ONLY: SING
        CHARACTER, INTENT(IN) :: TRANSA
        CHARACTER, INTENT(IN) :: TRANSB
        INTEGER, INTENT(IN) :: M, N, K
        REAL(KIND=SING), INTENT(IN) :: ALPHA
        REAL(KIND=SING), INTENT(IN), DIMENSION(LDA,*) :: A
        INTEGER, INTENT(IN) :: LDA
        REAL(KIND=SING), INTENT(IN), DIMENSION(LDB,*) :: B
        INTEGER, INTENT(IN) :: LDB
        REAL(KIND=SING), INTENT(IN) :: BETA
        REAL(KIND=SING), INTENT(IN), DIMENSION(LDC,*) :: C
        INTEGER, INTENT(IN) :: LDC
    END SUBROUTINE SGEMM
END INTERFACE GEMM

CONTAINS

! --------------------------------------------------------------------------------------------------

!> @brief Evaluate the activation function at the given point.
!> @param[in] X the input point
!> @return NONLINEARITY the value of the activation function at this point

ELEMENTAL REAL(KIND=NNP) FUNCTION NONLINEARITY(X)
    REAL(KIND=NNP), INTENT(IN) :: X

    ! Evaluate the activation function (currently hard-coded to the hyperbolic tangent)
    NONLINEARITY = TANH(X)
END FUNCTION NONLINEARITY

! --------------------------------------------------------------------------------------------------

!> @brief Evaluate the derivative of the activation function at the given point.
!> @param[in] X the input point
!> @return NONLINEARITY_TL the value of the activation function at this point

ELEMENTAL REAL(KIND=NNP) FUNCTION NONLINEARITY_TL(X)
    REAL(KIND=NNP), INTENT(IN) :: X

    ! Evaluate the derivative of the activation function (currently hard-coded to the hyperbolic
    ! tangent)
    NONLINEARITY_TL = 1.0_NNP - TANH(X)**2.0_NNP
END FUNCTION NONLINEARITY_TL

! --------------------------------------------------------------------------------------------------

!> @brief Evaluate the neural network for the given input array, of batch size KLON.
!> @param[in] NLON the size of the batch of input vectors (variable name copied from IFS)
!> @param[in] X the batch of input vectors
!> @param[out] Y the batch of output vectors

SUBROUTINE NN(KLON, X, Y)
    INTEGER,        INTENT(IN)  :: KLON
    REAL(KIND=NNP), INTENT(IN)  :: X(:,:)
    REAL(KIND=NNP), INTENT(OUT) :: Y(:,:)

    REAL(KIND=NNP), ALLOCATABLE :: T1(:,:), T2(:,:)
    INTEGER :: I
    
    ALLOCATE(T1(NWIDTH,KLON), T2(NWIDTH,KLON))
  
    ! Input layer
    T1 = INPUT_B
    CALL GEMM('N', 'N', &
        & NWIDTH, KLON, NINP, &
        & 1.0_NNP, &
        & INPUT, NWIDTH, &
        & X, NINP, &
        & 1._NNP, &
        & T1, NWIDTH)
    T2 = NONLINEARITY(T1)
  
    ! Hidden layers
    DO I = 1, NHIDDEN-1
        T1 = HIDDEN_B(:,:,I)
        CALL GEMM('N', 'N', &
            & NWIDTH, KLON, NWIDTH, &
            & 1.0_NNP, &
            & HIDDEN(:,:,I), NWIDTH, &
            & T2, NWIDTH, &
            & 1._NNP, &
            & T1, NWIDTH)
        T2 = NONLINEARITY(T1)     
    END DO
  
    ! Output layer
    Y = OUTPUT_B
    CALL GEMM('N', 'N', &
        & NOUT, KLON, NWIDTH, &
        & 1.0_NNP, &
        & OUTPUT, NOUT, &
        & T2, NWIDTH, &
        & 1._NNP, &
        & Y, NOUT)

END SUBROUTINE NN

! --------------------------------------------------------------------------------------------------

!> @brief Evaluate the tangent-linear of the neural network applied to an increment about a point X,
!> both given as batches of size KLON. The result is the evolved increment, and the evolved
!> nonlinear trajectory is given as a side-result.
!> @param[in] NLON the size of the batch of input vectors (variable name copied from IFS)
!> @param[in] X the batch of input vector linearisation points
!> @param[in] DX the batch of input perturbations
!> @param[out] Y the batch of output nonlinear trajectory evaluations
!> @param[out] DY the batch of output perturbations

SUBROUTINE NN_TL(KLON, X, DX, Y, DY)
    INTEGER,        INTENT(IN)  :: KLON
    REAL(KIND=NNP), INTENT(IN)  :: X(:,:), DX(:,:)
    REAL(KIND=NNP), INTENT(OUT) :: Y(:,:), DY(:,:)

    REAL(KIND=NNP), ALLOCATABLE :: T1(:,:), T2(:,:), T_NL(:,:)
    INTEGER :: I

    ALLOCATE(T1(NWIDTH,KLON), T2(NWIDTH,KLON), T_NL(NWIDTH,KLON))

    ! Weight matrix times input perturbation
    CALL GEMM('N', 'N', &
        & NWIDTH, KLON, NINP, &
        & 1.0_NNP, &
        & INPUT, NWIDTH, &
        & DX, NINP, &
        & 0.0_NNP, &
        & T1, NWIDTH)

    ! Nonlinear "trajectory"
    T_NL = INPUT_B
    CALL GEMM('N', 'N', &
        & NWIDTH, KLON, NINP, &
        & 1.0_NNP, &
        & INPUT, NWIDTH, &
        & X, NINP, &
        & 1.0_NNP, &
        & T_NL, NWIDTH)

    T2 = NONLINEARITY_TL(T_NL) * T1

    ! Hidden layers
    DO I = 1, NHIDDEN - 1
        ! Weight matrix times input perturbation
        CALL GEMM('N', 'N', &
            & NWIDTH, KLON, NWIDTH, &
            & 1.0_NNP, &
            & HIDDEN(:,:,I), NWIDTH, &
            & T2, NWIDTH, &
            & 0.0_NNP, &
            & T1, NWIDTH)

        ! Nonlinear "trajectory"
        T2 = NONLINEARITY(T_NL)
        T_NL = HIDDEN_B(:,:,I)
        CALL GEMM('N', 'N', &
            & NWIDTH, KLON, NWIDTH, &
            & 1.0_NNP, &
            & HIDDEN(:,:,I), NWIDTH, &
            & T2, NWIDTH, &
            & 1.0_NNP, &
            & T_NL, NWIDTH)

        T2 = NONLINEARITY_TL(T_NL) * T1
    END DO

    ! Output layer (no activation function)
    CALL GEMM('N', 'N', &
        & NOUT, KLON, NWIDTH, &
        & 1.0_NNP, &
        & OUTPUT, NOUT, &
        & T2, NWIDTH, &
        & 0.0_NNP, &
        & DY, NOUT)

    ! Nonlinear "trajectory"
    T2 = NONLINEARITY(T_NL)
    Y = OUTPUT_B
    CALL GEMM('N', 'N', &
        & NOUT, KLON, NWIDTH, &
        & 1.0_NNP, &
        & OUTPUT, NOUT, &
        & T2, NWIDTH, &
        & 1.0_NNP, &
        & Y, NOUT)

END SUBROUTINE NN_TL

! --------------------------------------------------------------------------------------------------

!> @brief Evaluate the adjoint of the neural network applied to a gradient about a point X, both
!> given as batches of size KLON. The result is a backwards-integrated gradient.
!> @param[in] NLON the size of the batch of input vectors (variable name copied from IFS)
!> @param[in] X the batch of input vector linearisation points
!> @param[in] GRADY the batch of input gradients
!> @param[out] GRADX the batch of output gradients

SUBROUTINE NN_AD(KLON, X, GRADY, GRADX)
    INTEGER,        INTENT(IN)  :: KLON
    REAL(KIND=NNP), INTENT(IN)  :: X(:,:), GRADY(:,:)
    REAL(KIND=NNP), INTENT(OUT) :: GRADX(:,:)

    REAL(KIND=NNP), ALLOCATABLE :: T1(:,:), T2(:,:), T_NL(:,:)
    INTEGER :: I, J

    ALLOCATE(T1(NWIDTH,KLON), T2(NWIDTH,KLON), T_NL(NWIDTH,KLON))

    ! Output layer (no activation function)
    CALL GEMM('T', 'N', &
        & NWIDTH, KLON, NOUT, &
        & 1.0_NNP, &
        & OUTPUT, NOUT, &
        & GRADY, NOUT, &
        & 0.0_NNP, &
        & T1, NWIDTH)

    ! Hidden layers
    DO I = NHIDDEN-1, 1, -1
        ! Compute nonlinear trajectory up to this hidden layer
        T_NL = INPUT_B
        CALL GEMM('N', 'N', &
            & NWIDTH, KLON, NINP, &
            & 1.0_NNP, &
            & INPUT, NWIDTH, &
            & X, NINP, &
            & 1.0_NNP, &
            & T_NL, NWIDTH)

        DO J = 1, I
           T2 = NONLINEARITY(T_NL)
           T_NL = HIDDEN_B(:,:,J)
           CALL GEMM('N', 'N', &
               & NWIDTH, KLON, NWIDTH, &
               & 1.0_NNP, &
               & HIDDEN(:,:,J), NWIDTH, &
               & T2, NWIDTH, &
               & 1.0_NNP, &
               & T_NL, NWIDTH)
        END DO

        T2 = NONLINEARITY_TL(T_NL) * T1
        CALL GEMM('T', 'N', &
            & NWIDTH, KLON, NWIDTH, &
            & 1.0_NNP, &
            & HIDDEN(:,:,I), NWIDTH, &
            & T2, NWIDTH, &
            & 0.0_NNP, &
            & T1, NWIDTH)
    END DO

    T_NL = INPUT_B
    CALL GEMM('N', 'N', &
        & NWIDTH, KLON, NINP, &
        & 1.0_NNP, &
        & INPUT, NWIDTH, &
        & X, NINP, &
        & 1.0_NNP, &
        & T_NL, NWIDTH)

    ! Input layer
    T2 = NONLINEARITY_TL(T_NL) * T1
    CALL GEMM('T', 'N', &
        & NINP, KLON, NWIDTH, &
        & 1.0_NNP, &
        & INPUT, NWIDTH, &
        & T2, NWIDTH, &
        & 0.0_NNP, &
        & GRADX, NINP)
END SUBROUTINE NN_AD

! --------------------------------------------------------------------------------------------------

END MODULE NOGWDNN_MOD

