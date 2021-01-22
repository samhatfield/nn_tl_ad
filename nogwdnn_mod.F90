MODULE NOGWDNN_MOD

USE PARKIND1
USE YONOGWDNN

IMPLICIT NONE

PUBLIC :: GEMV
INTERFACE GEMV
   SUBROUTINE DGEMV(TRANS, M, N, ALPHA, A, LDA, X, INCX, BETA, Y, INCY)
       USE PARKIND1, ONLY: DOUB
       CHARACTER, INTENT(IN) :: TRANS
       INTEGER, INTENT(IN) :: M, N
       REAL(KIND=DOUB), INTENT(IN) :: ALPHA
       REAL(KIND=DOUB), INTENT(IN), DIMENSION(LDA,*) :: A
       INTEGER, INTENT(IN) :: LDA
       REAL(KIND=DOUB), INTENT(IN), DIMENSION(*) :: X
       INTEGER, INTENT(IN) :: INCX
       REAL(KIND=DOUB), INTENT(IN) :: BETA
       REAL(KIND=DOUB), INTENT(INOUT), DIMENSION(*) :: Y
       INTEGER, INTENT(IN) :: INCY
   END SUBROUTINE DGEMV
   SUBROUTINE SGEMV(TRANS, M, N, ALPHA, A, LDA, X, INCX, BETA, Y, INCY)
       USE PARKIND1, ONLY: SING
       CHARACTER, INTENT(IN) :: TRANS
       INTEGER, INTENT(IN) :: M, N
       REAL(KIND=SING), INTENT(IN) :: ALPHA
       REAL(KIND=SING), INTENT(IN), DIMENSION(LDA,*) :: A
       INTEGER, INTENT(IN) :: LDA
       REAL(KIND=SING), INTENT(IN), DIMENSION(*) :: X
       INTEGER, INTENT(IN) :: INCX
       REAL(KIND=SING), INTENT(IN) :: BETA
       REAL(KIND=SING), INTENT(INOUT), DIMENSION(*) :: Y
       INTEGER, INTENT(IN) :: INCY
   END SUBROUTINE SGEMV
END INTERFACE GEMV

CONTAINS

! ---------------------------------------------------------------------------------

ELEMENTAL REAL(KIND=NNP) FUNCTION NONLINEARITY(X)
  REAL(KIND=NNP), INTENT(IN) :: X

  IF (LTANH) THEN
        NONLINEARITY = TANH(X)
     RETURN
  END IF
END FUNCTION NONLINEARITY

! ---------------------------------------------------------------------------------

ELEMENTAL REAL(KIND=NNP) FUNCTION NONLINEARITY_TL(X)
    REAL(KIND=NNP), INTENT(IN) :: X

    IF (LTANH) THEN
        NONLINEARITY_TL = 1.0_NNP - TANH(X)**2.0_NNP
        RETURN
    END IF
END FUNCTION NONLINEARITY_TL

! ---------------------------------------------------------------------------------

SUBROUTINE NN(X,Y)

  REAL(KIND=NNP), INTENT(IN) :: X(:)
  REAL(KIND=NNP), INTENT(OUT) :: Y(:)
  REAL(KIND=NNP), ALLOCATABLE :: T1(:),T2(:)

  INTEGER :: I
  
  ALLOCATE(T1(NWIDTH),T2(NWIDTH))


  !FIRST LAYER
  T1 = INPUT_B
  !IF (ERRORCHECK(T1)) PRINT*,0,T1
  CALL GEMV('N',NWIDTH,NINP,1.0_NNP,INPUT,NWIDTH,&
       & X,1,1._NNP,T1,1)
  !IF (ERRORCHECK(T1)) PRINT*,0.5,T1
  T2 = NONLINEARITY(T1)
  !IF (ERRORCHECK(T2)) PRINT*,1,T2
  !print*,T2


  !More hidden layers
  DO I = 1,NHIDDEN-1
     T1 = HIDDEN_B(:,I)
     !IF (ERRORCHECK(T1)) PRINT*,I+0.25,T1
     CALL GEMV('N',NWIDTH,NWIDTH,1.0_NNP,HIDDEN(:,:,I),NWIDTH,&
          & T2,1,1._NNP,T1,1)
     !IF (ERRORCHECK(T1)) PRINT*,I+0.5,T1
     T2 = NONLINEARITY(T1)     
     !IF (ERRORCHECK(T2)) PRINT*,I+1,T2
     !print*,T2
  END DO

  Y = OUTPUT_B
  !IF (ERRORCHECK(Y)) PRINT*,NHIDDEN+0.25,Y
  CALL GEMV('N',NOUT,NWIDTH,1.0_NNP,OUTPUT,NOUT,&
       & T2,1,1._NNP,Y,1)
  !IF (ERRORCHECK(Y)) PRINT*,NHIDDEN+0.5,Y
  !print*,Y

END SUBROUTINE NN

! ---------------------------------------------------------------------------------

SUBROUTINE NN_TL(X, DX, Y, DY)
    REAL(KIND=NNP), INTENT(IN) :: X(:), DX(:)
    REAL(KIND=NNP), INTENT(OUT) :: Y(:), DY(:)

    REAL(KIND=NNP), ALLOCATABLE :: T1(:), T2(:), T_NL(:)
    INTEGER :: I

    ALLOCATE(T1(NWIDTH), T2(NWIDTH), T_NL(NWIDTH))

    ! Weight matrix times input perturbation
    CALL GEMV('N', &
        & NWIDTH, NINP, &
        & 1.0_NNP, &
        & INPUT, NWIDTH, &
        & DX, 1, &
        & 0.0_NNP, &
        & T1, 1)

    ! Nonlinear "trajectory"
    T_NL = INPUT_B
    CALL GEMV('N', &
        & NWIDTH, NINP, &
        & 1.0_NNP, &
        & INPUT, NWIDTH, &
        & X, 1, &
        & 1.0_NNP, &
        & T_NL, 1)

    T2 = NONLINEARITY_TL(T_NL) * T1

    ! Hidden layers
    DO I = 1, NHIDDEN - 1
        ! Weight matrix times input perturbation
        CALL GEMV('N', &
            & NWIDTH, NWIDTH, &
            & 1.0_NNP, &
            & HIDDEN(:,:,I), NWIDTH, &
            & T2, 1, &
            & 0.0_NNP, &
            & T1, 1)

        ! Nonlinear "trajectory"
        T2 = NONLINEARITY(T_NL)
        T_NL = HIDDEN_B(:,I)
        CALL GEMV('N', &
            & NWIDTH, NWIDTH, &
            & 1.0_NNP, &
            & HIDDEN(:,:,I), NWIDTH, &
            & T2, 1, &
            & 1.0_NNP, &
            & T_NL, 1)

        T2 = NONLINEARITY_TL(T_NL) * T1
    END DO

    ! Output layer (no activation function)
    CALL GEMV('N', &
        & NOUT, NWIDTH, &
        & 1.0_NNP, &
        & OUTPUT, NOUT, &
        & T2, 1, &
        & 0.0_NNP, &
        & DY, 1)

    ! Nonlinear "trajectory"
    T2 = NONLINEARITY(T_NL)
    Y = OUTPUT_B
    T_NL = HIDDEN_B(:,I)
    CALL GEMV('N', &
        & NOUT, NWIDTH, &
        & 1.0_NNP, &
        & OUTPUT, NOUT, &
        & T2, 1, &
        & 1.0_NNP, &
        & Y, 1)
END SUBROUTINE NN_TL

! ---------------------------------------------------------------------------------

SUBROUTINE NN_AD(X, GRADY, GRADX)
    REAL(KIND=NNP), INTENT(IN) :: X(:), GRADY(:)
    REAL(KIND=NNP), INTENT(OUT) :: GRADX(:)

    REAL(KIND=NNP), ALLOCATABLE :: T1(:), T2(:), T_NL(:)
    INTEGER :: I, J

    ALLOCATE(T1(NWIDTH), T2(NWIDTH), T_NL(NWIDTH))

    ! Output layer (no activation function)
    CALL GEMV('T', &
        & NOUT, NWIDTH, &
        & 1.0_NNP, &
        & OUTPUT, NOUT, &
        & GRADY, 1, &
        & 0.0_NNP, &
        & T1, 1)

    ! Hidden layers
    DO I = NHIDDEN-1, 1, -1
        ! Compute nonlinear trajectory up to this hidden layer
        T_NL = INPUT_B
        CALL GEMV('N', &
            & NWIDTH, NINP, &
            & 1.0_NNP, &
            & INPUT, NWIDTH, &
            & X, 1, &
            & 1.0_NNP, &
            & T_NL, 1)

        DO J = 1, I
           T2 = NONLINEARITY(T_NL)
           T_NL = HIDDEN_B(:,J)
           CALL GEMV('N', &
               & NWIDTH, NWIDTH, &
               & 1.0_NNP, &
               & HIDDEN(:,:,J), NWIDTH, &
               & T2, 1, &
               & 1.0_NNP, &
               & T_NL, 1)
        END DO

        T2 = NONLINEARITY_TL(T_NL) * T1
        CALL GEMV('T', &
            & NWIDTH, NWIDTH, &
            & 1.0_NNP, &
            & HIDDEN(:,:,I), NWIDTH, &
            & T2, 1, &
            & 0.0_NNP, &
            & T1, 1)
    END DO

    T_NL = INPUT_B
    CALL GEMV('N', &
        & NWIDTH, NINP, &
        & 1.0_NNP, &
        & INPUT, NWIDTH, &
        & X, 1, &
        & 1.0_NNP, &
        & T_NL, 1)

    ! Input layer
    T2 = NONLINEARITY_TL(T_NL) * T1
    CALL GEMV('T', &
        & NWIDTH, NINP, &
        & 1.0_NNP, &
        & INPUT, NWIDTH, &
        & T2, 1, &
        & 0.0_NNP, &
        & GRADX, 1)
END SUBROUTINE NN_AD

END MODULE NOGWDNN_MOD
