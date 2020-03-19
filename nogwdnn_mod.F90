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
  CALL DGEMM('N','N',NWIDTH,1,NINP,1.0_JPRM,INPUT,NWIDTH,&
       & X,NINP,1._JPRM,T1,NWIDTH)
  !IF (ERRORCHECK(T1)) PRINT*,0.5,T1
  T2 = NONLINEARITY(T1)
  !IF (ERRORCHECK(T2)) PRINT*,1,T2
  !print*,T2

  !More hidden layers
  DO I = 1,NHIDDEN-1
     T1 = HIDDEN_B(:,I)
     !IF (ERRORCHECK(T1)) PRINT*,I+0.25,T1
     CALL DGEMM('N','N',NWIDTH,1,NWIDTH,1.0_JPRM,HIDDEN(:,:,I),NWIDTH,&
          & T2,NWIDTH,1._JPRM,T1,NWIDTH)
     !IF (ERRORCHECK(T1)) PRINT*,I+0.5,T1
     T2 = NONLINEARITY(T1)     
     !IF (ERRORCHECK(T2)) PRINT*,I+1,T2
     !print*,T2
  END DO

  Y = OUTPUT_B
  !IF (ERRORCHECK(Y)) PRINT*,NHIDDEN+0.25,Y
  CALL DGEMM('N','N',NOUT,1,NWIDTH,1.0_JPRM,OUTPUT,NOUT,&
       & T2,NWIDTH,1._JPRM,Y,NOUT)
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
    CALL DGEMM('N', 'N', &
        & NWIDTH, 1, NINP, &
        & 1.0_JPRM, &
        & INPUT, NWIDTH, &
        & DX, NINP, &
        & 0.0_JPRM, &
        & T1, NWIDTH)

    ! Nonlinear "trajectory"
    T_NL = INPUT_B
    CALL DGEMM('N', 'N', &
        & NWIDTH, 1, NINP, &
        & 1.0_JPRM, &
        & INPUT, NWIDTH, &
        & X, NINP, &
        & 1.0_JPRM, &
        & T_NL, NWIDTH)

    T2 = NONLINEARITY_TL(T_NL) * T1

    ! Hidden layers
    DO I = 1, NHIDDEN - 1
        ! Weight matrix times input perturbation
        CALL DGEMM('N', 'N', &
            & NWIDTH, 1, NWIDTH, &
            & 1.0_JPRM, &
            & HIDDEN(:,:,I), NWIDTH, &
            & T2, NWIDTH, &
            & 0.0_JPRM, &
            & T1, NWIDTH)

        ! Nonlinear "trajectory"
        T2 = NONLINEARITY(T_NL)
        T_NL = HIDDEN_B(:,I)
        CALL DGEMM('N', 'N', &
            & NWIDTH, 1, NWIDTH, &
            & 1.0_JPRM, &
            & HIDDEN(:,:,I), NWIDTH, &
            & T2, NWIDTH, &
            & 1.0_JPRM, &
            & T_NL, NWIDTH)

        T2 = NONLINEARITY_TL(T_NL) * T1
    END DO

    ! Output layer (no activation function)
    CALL DGEMM('N', 'N', &
        & NOUT, 1, NWIDTH, &
        & 1.0_JPRM, &
        & OUTPUT, NOUT, &
        & T2, NWIDTH, &
        & 0.0_JPRM, &
        & DY, NOUT)

    ! Nonlinear "trajectory"
    T2 = NONLINEARITY(T_NL)
    Y = OUTPUT_B
    T_NL = HIDDEN_B(:,I)
    CALL DGEMM('N', 'N', &
        & NOUT, 1, NWIDTH, &
        & 1.0_JPRM, &
        & OUTPUT, NOUT, &
        & T2, NWIDTH, &
        & 1.0_JPRM, &
        & Y, NOUT)
END SUBROUTINE NN_TL

! ---------------------------------------------------------------------------------

SUBROUTINE NN_AD(X, GRADY, GRADX)
    REAL(KIND=JPRM), INTENT(IN) :: X(:), GRADY(:)
    REAL(KIND=JPRM), INTENT(OUT) :: GRADX(:)

    REAL(KIND=JPRM), ALLOCATABLE :: T1(:), T2(:), T_NL(:)
    INTEGER :: I, J

    ALLOCATE(T1(NWIDTH), T2(NWIDTH), T_NL(NWIDTH))

    ! Output layer (no activation function)
    CALL DGEMM('T', 'N', &
        & NWIDTH, 1, NOUT, &
        & 1.0_JPRM, &
        & OUTPUT, NOUT, &
        & GRADY, NOUT, &
        & 0.0_JPRM, &
        & T1, NWIDTH)

    ! Hidden layers
    DO I = NHIDDEN-1, 1, -1
        ! Compute nonlinear trajectory up to this hidden layer
        T_NL = INPUT_B
        CALL DGEMM('N', 'N', &
            & NWIDTH, 1, NINP, &
            & 1.0_JPRM, &
            & INPUT, NWIDTH, &
            & X, NINP, &
            & 1.0_JPRM, &
            & T_NL, NWIDTH)

        DO J = 1, I
           T2 = NONLINEARITY(T_NL)
           T_NL = HIDDEN_B(:,J)
           CALL DGEMM('N', 'N', &
               & NWIDTH, 1, NWIDTH, &
               & 1.0_JPRM, &
               & HIDDEN(:,:,J), NWIDTH, &
               & T2, NWIDTH, &
               & 1.0_JPRM, &
               & T_NL, NWIDTH)
        END DO

        T2 = NONLINEARITY_TL(T_NL) * T1
        CALL DGEMM('T', 'N', &
            & NWIDTH, 1, NWIDTH, &
            & 1.0_JPRM, &
            & HIDDEN(:,:,I), NWIDTH, &
            & T2, NWIDTH, &
            & 0.0_JPRM, &
            & T1, NWIDTH)
    END DO

    T_NL = INPUT_B
    CALL DGEMM('N', 'N', &
        & NWIDTH, 1, NINP, &
        & 1.0_JPRM, &
        & INPUT, NWIDTH, &
        & X, NINP, &
        & 1.0_JPRM, &
        & T_NL, NWIDTH)

    ! Input layer
    T2 = NONLINEARITY_TL(T_NL) * T1
    CALL DGEMM('T', 'N', &
        & NINP, 1, NWIDTH, &
        & 1.0_JPRM, &
        & INPUT, NWIDTH, &
        & T2, NWIDTH, &
        & 0.0_JPRM, &
        & GRADX, NINP)
END SUBROUTINE NN_AD

END MODULE NOGWDNN_MOD