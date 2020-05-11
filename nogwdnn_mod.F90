MODULE NOGWDNN_MOD

USE PARKIND1, ONLY: JPRM
USE YONOGWDNN

IMPLICIT NONE

CONTAINS

! ---------------------------------------------------------------------------------

ELEMENTAL REAL(KIND=JPRM) FUNCTION NONLINEARITY(X)
  REAL(KIND=JPRM), INTENT(IN) :: X

  IF (LTANH) THEN
        NONLINEARITY = TANH(X)
     RETURN
  END IF
END FUNCTION NONLINEARITY

! ---------------------------------------------------------------------------------

ELEMENTAL REAL(KIND=JPRM) FUNCTION NONLINEARITY_TL(X)
    REAL(KIND=JPRM), INTENT(IN) :: X

    IF (LTANH) THEN
        NONLINEARITY_TL = 1.0_JPRM - TANH(X)**2.0_JPRM
        RETURN
    END IF
END FUNCTION NONLINEARITY_TL

! ---------------------------------------------------------------------------------

SUBROUTINE NN(X,Y)

  REAL(KIND=JPRM), INTENT(IN) :: X(:)
  REAL(KIND=JPRM), INTENT(OUT) :: Y(:)
  REAL(KIND=JPRM), ALLOCATABLE :: T1(:),T2(:)

  INTEGER :: I
  
  ALLOCATE(T1(NWIDTH),T2(NWIDTH))
  
  !FIRST LAYER
  T1 = INPUT_B
  !IF (ERRORCHECK(T1)) PRINT*,0,T1
  CALL DGEMV('N',NWIDTH,NINP,1.0_JPRM,INPUT,NWIDTH,&
       & X,1,1._JPRM,T1,1)
  !IF (ERRORCHECK(T1)) PRINT*,0.5,T1
  T2 = NONLINEARITY(T1)
  !IF (ERRORCHECK(T2)) PRINT*,1,T2
  !print*,T2

  !More hidden layers
  DO I = 1,NHIDDEN-1
     T1 = HIDDEN_B(:,I)
     !IF (ERRORCHECK(T1)) PRINT*,I+0.25,T1
     CALL DGEMV('N',NWIDTH,NWIDTH,1.0_JPRM,HIDDEN(:,:,I),NWIDTH,&
          & T2,1,1._JPRM,T1,1)
     !IF (ERRORCHECK(T1)) PRINT*,I+0.5,T1
     T2 = NONLINEARITY(T1)     
     !IF (ERRORCHECK(T2)) PRINT*,I+1,T2
     !print*,T2
  END DO

  Y = OUTPUT_B
  !IF (ERRORCHECK(Y)) PRINT*,NHIDDEN+0.25,Y
  CALL DGEMV('N',NOUT,NWIDTH,1.0_JPRM,OUTPUT,NOUT,&
       & T2,1,1._JPRM,Y,1)
  !IF (ERRORCHECK(Y)) PRINT*,NHIDDEN+0.5,Y
  !print*,Y

END SUBROUTINE NN

! ---------------------------------------------------------------------------------

SUBROUTINE NN_TL(X, DX, Y, DY)
    REAL(KIND=JPRM), INTENT(IN) :: X(:), DX(:)
    REAL(KIND=JPRM), INTENT(OUT) :: Y(:), DY(:)

    REAL(KIND=JPRM), ALLOCATABLE :: T1(:), T2(:), T_NL(:)
    INTEGER :: I

    ALLOCATE(T1(NWIDTH), T2(NWIDTH), T_NL(NWIDTH))

    ! Weight matrix times input perturbation
    CALL DGEMV('N', &
        & NWIDTH, NINP, &
        & 1.0_JPRM, &
        & INPUT, NWIDTH, &
        & DX, 1, &
        & 0.0_JPRM, &
        & T1, 1)

    ! Nonlinear "trajectory"
    T_NL = INPUT_B
    CALL DGEMV('N', &
        & NWIDTH, NINP, &
        & 1.0_JPRM, &
        & INPUT, NWIDTH, &
        & X, 1, &
        & 1.0_JPRM, &
        & T_NL, 1)

    T2 = NONLINEARITY_TL(T_NL) * T1

    ! Hidden layers
    DO I = 1, NHIDDEN - 1
        ! Weight matrix times input perturbation
        CALL DGEMV('N', &
            & NWIDTH, NWIDTH, &
            & 1.0_JPRM, &
            & HIDDEN(:,:,I), NWIDTH, &
            & T2, 1, &
            & 0.0_JPRM, &
            & T1, 1)

        ! Nonlinear "trajectory"
        T2 = NONLINEARITY(T_NL)
        T_NL = HIDDEN_B(:,I)
        CALL DGEMV('N', &
            & NWIDTH, NWIDTH, &
            & 1.0_JPRM, &
            & HIDDEN(:,:,I), NWIDTH, &
            & T2, 1, &
            & 1.0_JPRM, &
            & T_NL, 1)

        T2 = NONLINEARITY_TL(T_NL) * T1
    END DO

    ! Output layer (no activation function)
    CALL DGEMV('N', &
        & NOUT, NWIDTH, &
        & 1.0_JPRM, &
        & OUTPUT, NOUT, &
        & T2, 1, &
        & 0.0_JPRM, &
        & DY, 1)

    ! Nonlinear "trajectory"
    T2 = NONLINEARITY(T_NL)
    Y = OUTPUT_B
    T_NL = HIDDEN_B(:,I)
    CALL DGEMV('N', &
        & NOUT, NWIDTH, &
        & 1.0_JPRM, &
        & OUTPUT, NOUT, &
        & T2, 1, &
        & 1.0_JPRM, &
        & Y, 1)
END SUBROUTINE NN_TL

! ---------------------------------------------------------------------------------

SUBROUTINE NN_AD(X, GRADY, GRADX)
    REAL(KIND=JPRM), INTENT(IN) :: X(:), GRADY(:)
    REAL(KIND=JPRM), INTENT(OUT) :: GRADX(:)

    REAL(KIND=JPRM), ALLOCATABLE :: T1(:), T2(:), T_NL(:)
    INTEGER :: I, J

    ALLOCATE(T1(NWIDTH), T2(NWIDTH), T_NL(NWIDTH))

    ! Output layer (no activation function)
    CALL DGEMV('T', &
        & NWIDTH, NOUT, &
        & 1.0_JPRM, &
        & OUTPUT, NOUT, &
        & GRADY, 1, &
        & 0.0_JPRM, &
        & T1, 1)

    ! Hidden layers
    DO I = NHIDDEN-1, 1, -1
        ! Compute nonlinear trajectory up to this hidden layer
        T_NL = INPUT_B
        CALL DGEMV('N', &
            & NWIDTH, NINP, &
            & 1.0_JPRM, &
            & INPUT, NWIDTH, &
            & X, 1, &
            & 1.0_JPRM, &
            & T_NL, 1)

        DO J = 1, I
           T2 = NONLINEARITY(T_NL)
           T_NL = HIDDEN_B(:,J)
           CALL DGEMV('N', &
               & NWIDTH, NWIDTH, &
               & 1.0_JPRM, &
               & HIDDEN(:,:,J), NWIDTH, &
               & T2, 1, &
               & 1.0_JPRM, &
               & T_NL, 1)
        END DO

        T2 = NONLINEARITY_TL(T_NL) * T1
        CALL DGEMV('T', &
            & NWIDTH, NWIDTH, &
            & 1.0_JPRM, &
            & HIDDEN(:,:,I), NWIDTH, &
            & T2, 1, &
            & 0.0_JPRM, &
            & T1, 1)
    END DO

    T_NL = INPUT_B
    CALL DGEMV('N', &
        & NWIDTH, NINP, &
        & 1.0_JPRM, &
        & INPUT, NWIDTH, &
        & X, 1, &
        & 1.0_JPRM, &
        & T_NL, 1)

    ! Input layer
    T2 = NONLINEARITY_TL(T_NL) * T1
    CALL DGEMV('T', &
        & NINP, NWIDTH, &
        & 1.0_JPRM, &
        & INPUT, NWIDTH, &
        & T2, 1, &
        & 0.0_JPRM, &
        & GRADX, 1)
END SUBROUTINE NN_AD

END MODULE NOGWDNN_MOD
