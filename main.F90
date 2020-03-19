PROGRAM MAIN
    USE NOGWDNN_MOD, ONLY: NN, NN_TL
    USE YONOGWDNN
    USE PARKIND1, ONLY: JPRM

    IMPLICIT NONE
    
    REAL(JPRM) :: X1(NINP), X2(NINP), Y1(NOUT), Y2(NOUT), Y3(NOUT), DX(NINP), DY(NOUT), DIFF
    INTEGER :: I
    
    ALLOCATE(INPUT_B(NWIDTH))
    ALLOCATE(OUTPUT_B(NOUT))
    ALLOCATE(INPUT(NWIDTH, NINP))
    ALLOCATE(OUTPUT(NOUT, NWIDTH))
    ALLOCATE(HIDDEN_B(NWIDTH, NHIDDEN))
    ALLOCATE(HIDDEN(NWIDTH, NWIDTH, NHIDDEN))
    
    CALL INITIALIZE1(INPUT_B)
    CALL INITIALIZE1(OUTPUT_B)
    CALL INITIALIZE2(INPUT)
    CALL INITIALIZE2(OUTPUT)
    CALL INITIALIZE2(HIDDEN_B)
    CALL INITIALIZE3(HIDDEN)
    
    CALL INITIALIZE1(X1)
    
    DO I = 0, 7
        DX = 10.0_JPRM**(-REAL(I, JPRM))
        X2 = X1 + DX

        WRITE (*,'(A)') "====================================================="
        WRITE (*,'(A,ES7.1)') "Size of perturbation = ", DX(1)
        
        CALL NN(X1, Y1)
        CALL NN(X2, Y2)
        CALL NN_TL(X1, DX, Y3, DY)
        
        WRITE (*,'(A,10F21.17)') "LHS", Y2 - Y1
        WRITE (*,'(A,5F21.17)') "RHS", DY
        DIFF = ABS(Y2(1) - Y1(1) - DY(1))
        WRITE (*,'(A,I3)') "Difference of 1st element around digit number ", NINT(ABS(LOG10(DIFF)))
        WRITE (*,'(A)') "Nonlinear trajectory comparison (should be identical):"
        WRITE (*,'(5F21.17)') Y1
        WRITE (*,'(5F21.17)') Y3
    END DO
    
CONTAINS
    SUBROUTINE INITIALIZE1(X)
        REAL(JPRM), INTENT(OUT) :: X(:)
        
        INTEGER :: I
        
        DO I = 1, SIZE(X)
            X(I) = RANDN(0.0_JPRM, 1.0_JPRM)
        END DO
    END SUBROUTINE INITIALIZE1

    SUBROUTINE INITIALIZE2(X)
        REAL(JPRM), INTENT(OUT) :: X(:,:)
        
        INTEGER :: I, J
        
        DO I = 1, SIZE(X, 1)
            DO J = 1, SIZE(X, 2)
                X(I, J) = RANDN(0.0_JPRM, 1.0_JPRM)
            END DO
        END DO
    END SUBROUTINE INITIALIZE2

    SUBROUTINE INITIALIZE3(X)
        REAL(JPRM), INTENT(OUT) :: X(:,:,:)
        
        INTEGER :: I, J, K
        
        DO I = 1, SIZE(X, 1)
            DO J = 1, SIZE(X, 2)
                DO K = 1, SIZE(X, 3)
                    X(I, J, K) = RANDN(0.0_JPRM, 1.0_JPRM)
                END DO
            END DO
        END DO
    END SUBROUTINE INITIALIZE3

    FUNCTION RANDN(MEAN, STDEV)
        REAL(JPRM), INTENT(IN) :: MEAN, STDEV
        REAL(JPRM) :: U, V, RANDN
        REAL(JPRM) :: RAND(2)

        CALL RANDOM_NUMBER(RAND)

        ! Box-Muller method
        U = (-2.0_JPRM * LOG(RAND(1))) ** 0.5_JPRM
        V =   2.0_JPRM * 6.28318530718_JPRM * RAND(2)
        RANDN = MEAN + STDEV * U * SIN(V)
    END FUNCTION RANDN
END PROGRAM MAIN
