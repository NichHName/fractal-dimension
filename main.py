from experiments import (
    run_1d_convergence_test,
    run_2d_parameter_sweep,
    run_2d_convergence_test,
    generate_test_image
)

def main():
    while True:
        print("\n" + "="*40)
        print(" MAIN ENGINE")
        print("="*40)
        print("1. Run 1D Signal Convergence Test")
        print("2. Run 2D Parameter Sweep (Persistence, Scale, Octaves)")
        print("3. Run 2D Sample Convergence Test")
        print("4. Generate 2D Noise Image")
        print("Q. Quit")
        
        choice = input("\nSelect an option: ").strip().lower()

        if choice == '1':
            run_1d_convergence_test()
            
        elif choice == '2':
            param = input("Test parameter (p = persistence, s = scale, o = octaves): ").strip().lower()
            if param in ['p', 's', 'o']:
                run_2d_parameter_sweep(param)
            else:
                print("Invalid parameter selected.")
                
        elif choice == '3':
            # try:
            target_d = float(input("Enter theoretical target dimension (e.g., 2.0 to 3.0): "))
            # Gentle mathematical guardrail
            if target_d < 2.0 or target_d > 3.0:
                print("\n[!] Warning: For a 2D topographic surface, the fractal dimension mathematically must fall between 2.0 (smooth) and 3.0 (white noise).")
            
            n = int(input("Enter number of samples to test convergence (e.g., 30): "))
            
            run_2d_convergence_test(target_d=target_d, n=n)
                
            # except ValueError:
            #     print("Please enter valid numbers.")
                
        elif choice == '4':
            color = input("Choose color scheme (default, purple, minecraft): ").strip().lower()
            if color not in ['default', 'minecraft', 'purple']:
                color = 'default'
            generate_test_image(color=color)
            
        elif choice == 'q':
            print("Exiting.")
            break
            
        else:
            print("Invalid selection. Please try again.")

if __name__ == "__main__":
    main()