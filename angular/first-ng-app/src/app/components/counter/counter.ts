import { Component, signal } from '@angular/core';

@Component({
  selector: 'app-counter',
  imports: [],
  templateUrl: './counter.html',
  styleUrl: './counter.scss'
})
export class Counter {
  counter = signal(0);
  inc() {
    this.counter.update(x => x+1);
  }
  dec() {
    this.counter.update(x => x-1);
  }
  reset() {
    this.counter.set(0);
  }
}
