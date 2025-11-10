import { Component } from "@angular/core";
import { RouterOutlet } from "@angular/router";

// Import FormsModule for ngModel
import { FormsModule } from '@angular/forms';

// Import Angular Material components
import { MatCardModule } from '@angular/material/card';
import { MatFormFieldModule } from '@angular/material/form-field';
import { MatInputModule } from '@angular/material/input';
import { MatButtonModule } from '@angular/material/button';

@Component({
  selector: "app-root",
  standalone: true, // Explicitly add standalone flag for clarity
  imports: [
    RouterOutlet,
    FormsModule,
    MatCardModule,
    MatFormFieldModule,
    MatInputModule,
    MatButtonModule
  ],
  templateUrl: "./app.component.html",
  styleUrl: "./app.component.css",
})
export class AppComponent {
  // Properties for RL parameters
  learningRate: number = 0.0003;
  gamma: number = 0.99;
  epsilon: number = 0.2;

  // Property for training status
  status: string = "Parado";

  // Placeholder functions for buttons
  startTraining(): void {
    this.status = "Treinando...";
    console.log("Iniciando treinamento com:", { lr: this.learningRate, gamma: this.gamma, epsilon: this.epsilon });
  }

  pauseTraining(): void {
    this.status = "Pausado";
    console.log("Treinamento pausado.");
  }

  resetTraining(): void {
    this.status = "Parado";
    console.log("Treinamento reiniciado.");
  }
}
